import torch
import comfy.samplers
import comfy.model_management
import comfy.controlnet
import comfy.cldm.control_types
from comfy.cldm.control_types import UNION_CONTROLNET_TYPES
import folder_paths
from .cache import remove_cache
import numpy as np
from PIL import Image
import os
from .utils import *


class QwenImageIntegratedKSampler:

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        generation_mode = ['文生图', '图像编辑模式']
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "模型 - 扩散模型输入，用作图像生成的核心模型"}),
                "clip": ("CLIP", {"tooltip": "CLIP - CLIP模型，用于文本编码和条件生成"}),
                "vae": ("VAE", {"tooltip": "VAE - VAE模型输入，用于将潜空间解码为最终可见图像"}),
                "positive_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "placeholder": "正向提示词", "tooltip": "正向提示 - 描述期望图像元索的文本提示"}),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "placeholder": "负向提示词", "tooltip": "负向提示 - 描述要避免的图像元素的文本提示"}),
                "generation_mode": (generation_mode, {"tooltip": "生成模式 - 选择文生图或图生图模式"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 10, "tooltip": "批次数量 - 生成图像的数量"}),
                "width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8, "tooltip": "宽度(文生图-必填，图生图-填写缩放/填0不缩放)"}),
                "height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8, "tooltip": "高度(文生图-必填，图生图-填写缩放/填0不缩放)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "生成噪波的随机种。"}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 10000, "tooltip": "降噪的步数。"}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "CFG - 用于平衡随机性和提示词服从性。提高该值会使结果更加符合提示词，但过高会导致图像质量下降。"}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "采样器 - 采样算法，会影响结果质量、生成速度、风格样式。"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple", "tooltip": "调度器 - 控制逐渐移除噪波的方法。"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "降噪的强度，降低该值会保留原图的大部分内容从而实现图生图。"}),
            },
            "optional": {
                "image1": ("IMAGE", {"tooltip": "图像1（主图） - 参考图像1（主图），用于条件生成和潜空间编码。如果不传入，则文生图。"}),
                "image2": ("IMAGE", {"tooltip": "图像2 - 参考图像2，用于条件生成和潜空间编码"}),
                "image3": ("IMAGE", {"tooltip": "图像3 - 参考图像3，用于条件生成和潜空间编码"}),
                "image4": ("IMAGE", {"tooltip": "图像4 - 参考图像4，用于条件生成和潜空间编码"}),
                "image5": ("IMAGE", {"tooltip": "图像5 - 参考图像5，用于条件生成和潜空间编码"}),
                "latent": ("LATENT", {"tooltip": "Latent - 文生图、图生图（传入了主图）可不传，自动创建，如需使用ControlNet等可自行传入"}),
                "controlnet_data": ("CONTROL_NET_DATA", {"tooltip": "ControlNet 数据（可选） - 输入 ControlNet 集成加载器输出的数据包，直接应用 ControlNet 控制"}),
                "auraflow_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "AuraFlow移位 - 采样算法（AuraFlow） 移位Shift参数，影响速度和质量 (0-100)"}),
                "cfg_norm_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "CFGNorm强度 - CFG标准化强度，动态调整CFG指导强度 (0-100)"}),
                "enable_clean_gpu_memory": ("BOOLEAN", {"default": False, "tooltip": "清理显存占用 - 在采样/解码前后清理显存占用，以释放资源给其他应用"}),
                "enable_clean_cpu_memory_after_finish": ("BOOLEAN", {"default": False, "tooltip": "完成后清理内存 - 生成完成后清理CPU内存"}),
                "enable_sound_notification": ("BOOLEAN", {"default": False, "tooltip": "完成后播放声音 - 解码完成后播放通知声音以提醒用户"}),
                "auto_save_output_folder": ("STRING", {"default": "", "tooltip": "自动保存输出文件夹(留空不自动保存) - 留空则不执行保存"}),
                "output_filename_prefix": ("STRING", {"default": "auto_save", "tooltip": "输出文件名前缀 - 默认auto_save"}),
                "instruction": ("STRING", {"multiline": True, "default": "描述输入图像的关键特征（颜色、形状、大小、纹理、对象、背景），然后解释用户的文本指令应如何修改或改变图像。生成符合用户要求的新图像，同时在适当的情况下保持与原始输入的一致性。", "placeholder": "不建议修改", "tooltip": "指令 - 系统指令，用于指导参考图像的图像编辑"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "IMAGE")
    RETURN_NAMES = ("生成图像", "（可选）Latent", "缩放后原图")
    FUNCTION = "sample"
    CATEGORY = "sampling"
    # 注意语言文件中不能用@符号
    DESCRIPTION = "NAKU 专用 QWEN集成采样器 - K采样器，智能多模态采样器，支持文生图/图生图双模式，优化官方偏移问题，更遵从指令，图片缩放、可处理多张参考图、自动显存/内存管理、批量生成、自动保存、声音通知、AuraFlow优化、CFG标准化调节等全方位功能，不需要连那么多线啦~~~~"


    


    def sample(self, model, clip, vae, positive_prompt, negative_prompt, generation_mode, batch_size, width, height, seed, steps, cfg, sampler_name, scheduler, denoise=1.0, image1=None, image2=None, image3=None, image4=None, image5=None, latent=None, controlnet_data=None, auraflow_shift=0, cfg_norm_strength=0, enable_clean_gpu_memory=False, enable_clean_cpu_memory_after_finish=False, enable_sound_notification=False, instruction="", auto_save_output_folder="", output_filename_prefix="auto_save"):


        # Print start execution information
        print(f"开始执行采样任务......")
        print(f"种子: {seed}")
        print(f"步骤数: {steps}")
        print(f"CFG强度: {cfg}")
        print(f"降噪强度: {denoise}")
        print(f"采样器: {sampler_name}")
        print(f"调度器: {scheduler}")
        print(f"分辨率: {width} x {height}")
        print(f"生成模式: {generation_mode}")
        print(f"批量大小: {batch_size}")
        print(f"AuraFlow Shift: {auraflow_shift}")
        print(f"CFG规范化强度: {cfg_norm_strength}")
        print(f"正向提示词长度: {len(positive_prompt)}")
        print(f"负向提示词长度: {len(negative_prompt)}")
        print(f"指令长度: {len(instruction)}")
        print(f"自动保存文件夹: {auto_save_output_folder if auto_save_output_folder else '禁用'}")
        print(f"输出文件名前缀: {output_filename_prefix}")
        print(f"清理GPU内存: {enable_clean_gpu_memory}")
        print(f"结束后清理CPU内存: {enable_clean_cpu_memory_after_finish}")
        print(f"声音通知: {enable_sound_notification}")

        # Initialize scaled images
        image1_scaled = image1


        if auraflow_shift > 0:
            print(f"✨ 应用shift参数/Applying shift parameter: {auraflow_shift}")
            m = model.clone()
            import comfy.model_sampling
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST
            class ModelSamplingAdvanced(sampling_base, sampling_type):
                pass
            model_sampling = ModelSamplingAdvanced(m.model.model_config)
            model_sampling.set_parameters(shift=auraflow_shift, multiplier=1.0)
            m.add_object_patch("model_sampling", model_sampling)
            model = m
            print("shift参数已应用成功/Shift parameter applied successfully")

        if cfg_norm_strength > 0:
            print(f"应用强度: {cfg_norm_strength}")
            m = model.clone()
            def cfg_norm(args):
                cond_p = args['cond_denoised']
                pred_text_ = args["denoised"]
                norm_full_cond = torch.norm(cond_p, dim=1, keepdim=True)
                norm_pred_text = torch.norm(pred_text_, dim=1, keepdim=True)
                scale = (norm_full_cond / (norm_pred_text + 1e-8)).clamp(min=0.0, max=1.0)
                return pred_text_ * scale * cfg_norm_strength
            m.set_model_sampler_post_cfg_function(cfg_norm)
            model = m
            print("规范化已应用成功/Normalization applied successfully")




        

        if generation_mode == "图像编辑模式":

            if image1 is None:
                raise Exception("图像编辑模式必须至少输入一张图片，请输入图像1（主图）。")

            # Scale reference images if needed

            images_scaled = [image1, image2, image3, image4, image5]

            if width > 0 and height > 0:
                print(f"[图像缩放] 将图像缩放至 {width}x{height}")

                for i, img in enumerate(images_scaled):
                    if img is not None:
                        # try:
                            scaled_image, _, _, _, _ = image_scale_by_aspect_ratio('original', 1, 1, 'letterbox', 'lanczos', '8', 'max_size', (width, height), '#000000', img, None)
                            images_scaled[i] = scaled_image
                        # except Exception as e:
                        #  log(f"⚠️ [Image Scale] Cannot scale image {i+1} with shape {img.shape}: {e}")
                        #     images_scaled[i] = img
                    # else: None

                print("[图像缩放] 图像缩放完成")

            image1_scaled, image2_scaled, image3_scaled, image4_scaled, image5_scaled = images_scaled

            image_prompt, images_vl, llama_template, ref_latents = get_image_prompt(vae, image1_scaled, image2_scaled, image3_scaled, image4_scaled, image5_scaled, upscale_method="lanczos", crop="disabled", instruction=instruction)

            print("⏳ [提示词] 正在生成正向条件...")
            positive = prompt_encode(clip, positive_prompt, image_prompt=image_prompt, images_vl=images_vl, llama_template=llama_template, ref_latents=ref_latents)
            print("⏳ [提示词] 正在生成负向条件...")
            negative = prompt_encode(clip, negative_prompt, image_prompt=image_prompt, images_vl=images_vl, llama_template=llama_template, ref_latents=ref_latents)
            print("[提示词] 提示词条件生成完成")


        else:
            if width > 0 and height > 0:
                positive = prompt_encode(clip, positive_prompt)
                negative = prompt_encode(clip, negative_prompt)
            else:
                raise Exception("文生图必须输入宽高。")

        # Apply ControlNet if provided
        if controlnet_data is not None and len(controlnet_data) > 0:
            try:
                # 预先处理所有controlnet数据以避免重复操作
                for c_idx, c_data in enumerate(controlnet_data):
                    control_net = c_data["control_net"]
                    control_type = c_data["control_type"]
                    control_image = c_data["image"]
                    control_mask = c_data["mask"]
                    control_strength = c_data["strength"]
                    control_start_percent = c_data["start_percent"]
                    control_end_percent = c_data["end_percent"]

                    print(f"应用ControlNet {control_type} 强度: {control_strength} (第{c_idx+1}/{len(controlnet_data)}个)")

                    if control_strength > 0:

                        # 图像编辑-局部重绘 时使用缩放主图
                        if generation_mode == "图像编辑模式":
                            if control_type.lower() == "repaint" or control_type.lower() == "inpaint" or control_type.lower() == "inpainting" or control_type == "重绘" or control_type == "局部重绘":

                                if width > 0 and height > 0:
                                    scaled_control_image, scaled_control_mask, _, _, _ = image_scale_by_aspect_ratio('original', 1, 1, 'letterbox', 'lanczos', '8', 'max_size', (width, height), '#000000', control_image, control_mask)
                                    control_image = scaled_control_image
                                    control_mask = scaled_control_mask

                        extra_concat=[]

                        if control_net.concat_mask and control_mask is not None:
                            control_mask = 1.0 - control_mask.reshape((-1, 1, control_mask.shape[-2], control_mask.shape[-1]))
                            control_mask_apply = comfy.utils.common_upscale(control_mask, control_image.shape[2], control_image.shape[1], "bilinear", "center").round()
                            control_image = control_image * control_mask_apply.movedim(1, -1).repeat(1, 1, 1, control_image.shape[3])
                            extra_concat = [control_mask]
                            print(f"ControlNet {control_type}应用遮罩")

                        control_hint = control_image.movedim(-1,1)

                        # 为每个ControlNet创建独立的处理流程，避免累积内存
                        cnets = {}

                        # 应用ControlNet到正向条件
                        positive_c = []
                        for t in positive:
                            d = t[1].copy()
                            prev_cnet = d.get('control', None)
                            if prev_cnet in cnets:
                                c_net = cnets[prev_cnet]
                            else:
                                c_net = control_net.copy().set_cond_hint(control_hint, control_strength, (control_start_percent, control_end_percent), vae=vae, extra_concat=extra_concat)
                                c_net.set_previous_controlnet(prev_cnet)
                                cnets[prev_cnet] = c_net

                            d['control'] = c_net
                            d['control_apply_to_uncond'] = False
                            n = [t[0], d]
                            positive_c.append(n)

                        # 应用ControlNet到负向条件
                        negative_c = []
                        for t in negative:
                            d = t[1].copy()
                            prev_cnet = d.get('control', None)
                            if prev_cnet in cnets:
                                c_net = cnets[prev_cnet]
                            else:
                                # 对于负向条件，使用相同的已缓存ControlNet实例
                                c_net = cnets.get(prev_cnet, control_net.copy()).set_cond_hint(control_hint, control_strength, (control_start_percent, control_end_percent), vae=vae, extra_concat=extra_concat)
                                c_net.set_previous_controlnet(prev_cnet)
                                cnets[prev_cnet] = c_net

                            d['control'] = c_net
                            d['control_apply_to_uncond'] = False
                            n = [t[0], d]
                            negative_c.append(n)

                        positive = positive_c
                        negative = negative_c

                        print(f"ControlNet {control_type}应用成功")

                        # 清理临时变量以释放内存
                        del control_hint, cnets
                        if enable_clean_gpu_memory or c_idx < len(controlnet_data) - 1:  # 在处理多个ControlNet时，或启用清理时，都进行清理
                            comfy.model_management.cleanup_models()
                            comfy.model_management.soft_empty_cache()
                            gc.collect()
                    else:
                        print(f"⚠️ ControlNet {control_type}强度设置为0，不应用ControlNet")

                # 在所有ControlNet处理完成后进行最终清理
                gc.collect()
                comfy.model_management.soft_empty_cache()
            except Exception as e:
                raise Exception(f"⚠️ [ControlNet] ControlNet 应用失败: {e}")



        if latent is None:
            if generation_mode == "图像编辑模式" and image1_scaled is not None:
                samples = vae.encode(image1_scaled[:,:,:,:3])  # Use scaled image1 (image1_scaled), not original
                if batch_size > 1:
                    samples = samples.repeat((batch_size,) + ((1,) * (samples.ndim - 1)))
            else:
                samples = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
            latent = {"samples":samples}

        

        


        print("🚀 开始采样过程...")
        print(f"总步数: {steps}")

        if enable_clean_gpu_memory:
            print("预清理显存占用...")
            try:
                cleanGPUUsedForce()
                remove_cache('*')
            except ImportError:
                print("🔕 显存清理失败")
            print("预显存清理完成")
        else:
            # 即使未启用显式清理，也执行基本的内存管理
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            gc.collect()

        # 在采样前再次确保模型被正确加载到适当的设备上
        model = model.to(comfy.model_management.get_torch_device())

        latent_output = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=denoise)

        print("正在解码潜空间...")
        output_images = vae.decode(latent_output["samples"])
        if len(output_images.shape) == 5: #Combine batches
            output_images = output_images.reshape(-1, output_images.shape[-3], output_images.shape[-2], output_images.shape[-1])
        print("解码完成")

        # 解码后进行显存清理
        if enable_clean_gpu_memory:
            comfy.model_management.cleanup_models()
            comfy.model_management.soft_empty_cache()
            gc.collect()



        if auto_save_output_folder:
            try:
                import folder_paths
                output_filename_prefix = output_filename_prefix or "auto_save"
                if os.path.isabs(auto_save_output_folder):
                    full_output_folder = auto_save_output_folder
                else:
                    output_dir = folder_paths.get_output_directory()
                    full_output_folder = os.path.join(output_dir, auto_save_output_folder)

                if not os.path.exists(full_output_folder):
                    os.makedirs(full_output_folder, exist_ok=True)

                print(f"💾 [Auto Save] 自动保存文件至 【{full_output_folder}】")

                for batch_number, image in enumerate(output_images):
                    img = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
                    file = f"{output_filename_prefix}_{seed}_{batch_number:05}.png"
                    img.save(os.path.join(full_output_folder, file))

                print("[Auto Save] 自动保存文件成功")
            except ImportError:
                print("🔕 自动保存文件失败")


        if enable_clean_gpu_memory:
            print("后清理显存占用...")
            try:
                cleanGPUUsedForce()
                remove_cache('*')
            except ImportError:
                print("🔕 显存清理失败")
            print("后显存清理完成")

        if enable_clean_cpu_memory_after_finish:
            print("完成后清理CPU内存...")
            try:
                clean_ram(clean_file_cache=True, clean_processes=True, clean_dlls=True, retry_times=3)
            except Exception as e:
                print(f"🔕 RAM清理失败: {str(e)}")
            else:
                print("[Clean CPU Memory After Finish] RAM清理完成")

        if enable_sound_notification:
            try:
                import winsound
                import time
                # 播放快速紧凑的旋律：A4, C5, E5, G5, E5, G5，较短间隔使旋律连贯
                frequencies = [440, 523, 659, 784, 659, 784]
                for freq in frequencies:
                    winsound.Beep(freq, 150)
                    time.sleep(0.005)  # 更短间隔加快节奏
                print("[声音通知] 完成旋律播放")
            except ImportError:
                print("🔕 [声音通知] 该系统不支持声音通知")
            except Exception as e:
                print(f"🔕 [声音通知] 音频播放失败: {str(e)}")

        return (output_images, latent_output, image1_scaled)
        


    def set_shift(self, model, sigma_shift):
        """设置AuraFlow模型的shift参数"""
        import comfy.model_sampling

        model_sampling = model.get_model_object("model_sampling")
        if not model_sampling:
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST
            class ModelSamplingAdvanced(sampling_base, sampling_type):
                pass
            model_sampling = ModelSamplingAdvanced(model.model.model_config)

        model_sampling.set_parameters(shift=sigma_shift / 1000.0 * 100, multiplier=1000)
        model.add_object_patch("model_sampling", model_sampling)
        return model

class ExtraOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

            }
        }

    RETURN_TYPES = ("EXTRA_OPTIONS",)
    RETURN_NAMES = ("额外设定",)
    FUNCTION = "get_options"
    CATEGORY = "sampling"
    DESCRIPTION = "额外设定 - 高级参数设置"

    def get_options(self):
        options = {

        }
        return (options,)

class QwenImageControlNetIntegratedLoader:
    @classmethod
    def INPUT_TYPES(s):
        type_options = ["auto"] + list(UNION_CONTROLNET_TYPES.keys())
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "控制图像 - ControlNet 控制图像输入"}),
                "control_net_name": (folder_paths.get_filename_list("controlnet"), {"tooltip": "ControlNet模型 - 选择要使用的 ControlNet 模型"}),
                "control_type": (type_options, {"default": "auto", "tooltip": "控制类型 - ControlNet 的具体控制类型，如姿态、 边缘检测、深度等"}),
                "strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "控制强度 - ControlNet 对生成结果的影响强度"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "控制步数开始百分比 - ControlNet 效果开始应用的步数百分比"}),
                "end_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "控制步数结束百分比 - ControlNet 效果结束应用的步数百分比"}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "遮罩（可选） - Inpainting 遮罩，用于 ControlNet 区域控制"}),
                "controlnet_data": ("CONTROL_NET_DATA", {"tooltip": "ControlNet 数据（可选） - 输入 ControlNet 集成加载器输出的数据包，直接应用 ControlNet 控制"}),
            }
        }

    RETURN_TYPES = ("CONTROL_NET_DATA",)
    RETURN_NAMES = ("ControlNet 数据",)
    FUNCTION = "load_controlnet"
    CATEGORY = "conditioning/controlnet"
    DESCRIPTION = "NAKU 专用 QWEN-ControlNet 集成加载器 - 需要 ControlNet 时使用"

    def load_controlnet(self, image, control_net_name, control_type, strength, start_percent, end_percent, mask=None, controlnet_data=None):

        if strength > 0:
            if image is None:
                raise Exception("错误: 使用ControlNet必须传入控制图像。")
            
        if control_type.lower() == "repaint" or control_type.lower() == "inpaint" or control_type.lower() == "inpainting" or control_type == "重绘" or control_type == "局部重绘":
            if mask is None:
                raise Exception("错误: 使用局部重绘ControlNet必须传入控制遮罩。")

        # 加载 ControlNet
        controlnet = comfy.controlnet.load_controlnet(folder_paths.get_full_path_or_raise("controlnet", control_net_name))

        if controlnet is None:
            raise RuntimeError("错误: 控制器文件无效，不包含有效的controlnet模型。")

        # 设置控制类型
        controlnet = controlnet.copy()
        type_number = UNION_CONTROLNET_TYPES.get(control_type, -1)
        if type_number >= 0:
            controlnet.set_extra_arg("control_type", [type_number])
        else:
            controlnet.set_extra_arg("control_type", [])

        if controlnet_data is not None and len(controlnet_data) > 0:
            control_net_data_list = controlnet_data
        else:
            control_net_data_list = []

        # 创建数据对象
        control_net_data = {
            "control_net": controlnet,
            "control_type": control_type,
            "image": image,
            "mask": mask,
            "strength": strength,
            "start_percent": start_percent,
            "end_percent": end_percent
        }

        control_net_data_list.append(control_net_data)

        return (control_net_data_list,)

NODE_CLASS_MAPPINGS = {
    "QwenImageIntegratedKSampler": QwenImageIntegratedKSampler,
    "QwenImageControlNetIntegratedLoader": QwenImageControlNetIntegratedLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageIntegratedKSampler": "NAKU 专用 QWEN集成采样器",
    "QwenImageControlNetIntegratedLoader": "NAKU 专用 QWEN-ControlNet集成加载器",
}
