import argparse
import base64
import io
import time

import torch
import torchvision
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image
from upscaler.models.esrgan.modelling_esrgan import ESRGAN, ESRGANConfig

app = Flask(
    __name__,
    static_url_path='',
)
CORS(app)

BATCH_SIZE = 1

ESRGAN_TILE = 512
MODEL_NAME = "RealESRGAN_x4plus"
# MODEL_NAME = "RealESRGAN_x4plus_anime_6B"
# MODEL_NAME = "RealESRNet_x4plus"
# MODEL_NAME = "RealESRGAN_x2plus"
# MODEL_NAME = "pbaylies_paintings"
# MODEL_NAME = "ESRGAN_SRx4_DF2KOST_official-ff704c30"

esrgan_config = ESRGANConfig(
    model_name=MODEL_NAME,
    tile=ESRGAN_TILE,
)
esrgan = ESRGAN(esrgan_config, )


def to_binary(image, ):
    buffered = io.BytesIO()
    image.save(
        buffered,
        format='jpeg',
        quality=90,
    )
    buffered.seek(0)
    return buffered


def main(args, ):
    @app.route('/upscale', methods=['POST'])
    def upscale():
        print(
            "REQ",
            request.form,
        )
        time_start = time.time()
        # TODO: handle batching
        img_base64 = base64.b64decode(request.form['image'])

        img_base64_list = [img_base64]
        num_imgs = len(img_base64_list)

        img_tensor_list = []
        upscaled_img_list = []
        for img_idx, img_base64 in enumerate(img_base64_list):
            try:
                img_pil = Image.open(io.BytesIO(img_base64)).convert("RGB")
                img_tensor = torchvision.transforms.PILToTensor()(img_pil)
                img_tensor = img_tensor[None, :].float() / 255.

                img_tensor_list.append(img_tensor)

                if (img_idx + 1) % BATCH_SIZE == 0 or img_idx + 1 == num_imgs:
                    batch_img_tensor = torch.cat(
                        img_tensor_list,
                        dim=0,
                    )
                    batch_upscaled_img_tensor = esrgan.upscale(
                        batch_img_tensor, )

                    for img_idx, upsacled_img_tensor in enumerate(
                            batch_upscaled_img_tensor):
                        upscaled_img_list.append(
                            torchvision.transforms.ToPILImage()(
                                upsacled_img_tensor, ))

                    # TODO: handle batching
                    upscaled_img = upscaled_img_list[0]

                    print('Upscaled in %f seconds.' %
                          (time.time() - time_start))

                return send_file(
                    to_binary(upscaled_img),
                    mimetype='image/JPEG',
                )

            except Exception as e:
                print('Error in /upscale', e)
                return '', 500

    app.run(host='0.0.0.0', debug=False, port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='waifu2x upscale server.', )
    parser.add_argument(
        '-p',
        '--port',
        type=int,
        default=5001,
    )
    main(args=parser.parse_args(), )
