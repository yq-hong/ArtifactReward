import os
from io import BytesIO
import time
import base64
import PIL.Image
import configs
import openai
from openai import OpenAI, BadRequestError
import pathlib

media = pathlib.Path(__file__).parents[1] / "third_party"


def encode_image(image_path):
    _, file_extension = os.path.splitext(image_path)
    file_extension = file_extension.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }
    mime_type = mime_types.get(file_extension)
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_image, mime_type


def gpt4o(prompt, img_paths=None, temperature=0.7, n=1, top_p=1, max_tokens=1024,
          presence_penalty=0, frequency_penalty=0, logit_bias={}, model_version="gpt-4o-2024-05-13", eval_image=True):
    client = OpenAI(api_key=configs.OPENAI_KEY)

    imgs_url = []
    for i in range(len(img_paths)):
        base64_image, mime_type = encode_image(img_paths[i])
        imgs_url.append({"type": "image_url",
                         "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}})

    if eval_image:
        messages = [{"role": "system",
                     "content": [{"type": "text",
                                  "text": "You are a professional Vincennes image quality audit expert, please evaluate the image quality strictly according to the protocol."}]}]
    else:
        messages = []
    messages += [{"role": "user",
                  "content": [{"type": "text", "text": prompt}] + imgs_url}]

    num_attempts = 0
    while num_attempts < 5:
        num_attempts += 1
        try:
            response = client.chat.completions.create(model=model_version,
                                                      # gpt-4o-mini-2024-07-18, gpt-4o-2024-08-06, gpt-4o-2024-05-13
                                                      messages=messages,
                                                      temperature=temperature,
                                                      n=n,
                                                      top_p=top_p,
                                                      max_tokens=max_tokens,
                                                      presence_penalty=presence_penalty,
                                                      frequency_penalty=frequency_penalty,
                                                      logit_bias=logit_bias
                                                      )
            num_attempts = 5
            return [response.choices[i].message.content for i in range(n)]

        except BadRequestError as be:
            print(f"BadRequestError: {be}")
            continue
        except openai.RateLimitError as e:
            print("Resource Exhausted, wait for a minute to continue...")
            time.sleep(60)
            continue
        except Exception as e:
            print(f"OpenAI server offers this error: {e}")
            if num_attempts < 5:
                time.sleep(5)  # Wait for 5 seconds before the next attempt
            continue
