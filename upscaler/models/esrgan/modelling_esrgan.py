import glob
import logging
import os

import torch
import torchvision
from PIL import Image

from upscaler.models.esrgan.modelling_esrgan_utils import RRDBNet, RealESRGANer
# from gfpgan import GFPGANer

logging.basicConfig(
    format="%(levelname)s:%(message)s",
    level=logging.DEBUG,
)

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")


class ESRGANConfig:
    def __init__(
        self,
        model_name: str = "RealESRGAN_x4plus",
        net_scale_factor: int = 4,
        out_img_scale_factor: int = 4,
        tile: int = 512,
        tile_pad: int = 10,
        pre_img_pad: int = 0,
        face_enhance: bool = False,
        half: bool = False,
        alpha_upsampler: str = "realsrgan",
        **kwargs,
    ):
        """
        Configuration for ESRGAN models.
        """

        self.model_name = model_name
        self.net_scale_factor = net_scale_factor
        self.out_img_scale_factor = out_img_scale_factor
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_img_pad = pre_img_pad
        self.face_enhance = face_enhance
        self.half = half
        self.alpha_upsampler = alpha_upsampler

        available_model_name_list = [
            "RealESRGAN_x4plus",
            "RealESRGAN_x4plus_anime_6B",
            "RealESRNet_x4plus",
            "RealESRGAN_x2plus",
            "pbaylies_paintings",
            "ESRGAN_SRx4_DF2KOST_official-ff704c30",
        ]
        # assert model_name in available_model_name_list, (
        #     f"'{model_name}' not available."
        #     f"Use {' or '.join(available_model_name_list)}")

        available_alpha_upsamplers = [
            "realsrgan",
            "bicubic",
        ]
        assert alpha_upsampler in available_alpha_upsamplers, (
            f"'{alpha_upsampler}'' not available"
            f"Use {' or '.join(available_alpha_upsamplers)}")

        self.model_path = os.path.join(CACHE_DIR, f"{model_name}.pth")

        self.num_block_rrdb = 23
        if model_name == "RealESRGAN_x4plus_anime_6B" or model_name == "pbayles_portraits":
            self.num_block_rrdb = 6

        elif model_name == "RealESRGAN_x2plus":
            self.net_scale_factor = 2


class ESRGAN:
    def __init__(
        self,
        config: ESRGANConfig,
    ):
        self.config = config

        rrdbnet = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=self.config.num_block_rrdb,
            num_grow_ch=32,
            scale=self.config.net_scale_factor,
        )

        self.esrgan = RealESRGANer(
            scale=self.config.net_scale_factor,
            model_path=self.config.model_path,
            model=rrdbnet,
            tile=self.config.tile,
            tile_pad=self.config.tile_pad,
            pre_pad=self.config.pre_img_pad,
            half=self.config.half,
        )

        # if self.config.face_enhance:
        #     face_enhancer = GFPGANer(
        #         model_path=
        #         'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth',
        #         upscale=self.config.out_img_scale_factor,
        #         arch='clean',
        #         channel_multiplier=2,
        #         bg_upsampler=self.esrgan,
        #     )

    def upscale(
        self,
        img: torch.Tensor,
    ):
        # XXX NOT REALLY IMPLEMENTED
        if self.config.face_enhance:
            _, _, upscaled_img = self.face_enhancer.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )

        else:
            upscaled_img = self.esrgan.enhance(
                img,
                outscale=self.config.out_img_scale_factor,
            )

        return upscaled_img


if __name__ == '__main__':
    test_img_dir = "test_imgs"
    out_img_dir = "results"
    input_img_path_list = glob.glob(os.path.join(test_img_dir, "*"))

    model_name = "RealESRGAN_x4plus"
    # model_name = "RealESRGAN_x4plus_anime_6B"
    # model_name = "RealESRNet_x4plus"
    # model_name = "RealESRGAN_x2plus"
    # model_name = "pbaylies_paintings"
    # model_name = "ESRGAN_SRx4_DF2KOST_official-ff704c30"

    tile = 512

    batch_size = 1

    esrgan_config = ESRGANConfig(
        model_name=model_name,
        tile=tile,
    )

    esrgan = ESRGAN(esrgan_config, )

    img_tensor_list = []
    input_img_filename_list = []
    for img_idx, input_img_path in enumerate(input_img_path_list):
        input_img_filename = os.path.basename(input_img_path, )

        img_pil = Image.open(input_img_path).convert("RGB")
        img_tensor = torchvision.transforms.PILToTensor()(img_pil)
        img_tensor = img_tensor[None, :].float() / 255.

        img_tensor_list.append(img_tensor)
        input_img_filename_list.append(input_img_filename)

        if (img_idx + 1) % batch_size == 0 or img_idx + 1 == len(
                input_img_path_list):
            batch_img_tensor = torch.cat(
                img_tensor_list,
                dim=0,
            )
            try:
                batch_upscaled_img_tensor = esrgan.upscale(batch_img_tensor, )

                for img_idx, upsacled_img_tensor in enumerate(
                        batch_upscaled_img_tensor):
                    torchvision.transforms.ToPILImage()(
                        upsacled_img_tensor,
                    ).save(
                        os.path.join(
                            out_img_dir,
                            f"{input_img_filename_list[img_idx].split('.')[0]}_esrgan.jpg",
                        ))

                logging.info(f"{input_img_filename} upscaled!")

            except Exception as e:
                logging.error(f"FAILED! {e}")

            finally:
                img_tensor_list = []
                input_img_filename_list = []
