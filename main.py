import argparse
import os
import tqdm
import logging
from network import model_detect

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def __work_mode__(path: str):
    
    if os.path.isfile(path):  # Input is file
        return "file"
    if os.path.isdir(path):  # Input is dir
        return "dir"
    else:
        return "no"


def __save_image_file__(img, file_name, output_path, wmode):
    
    if wmode == "file":
        file_name_out = os.path.basename(output_path)
        if file_name_out == '':
            # Change file extension to png
            file_name = os.path.splitext(file_name)[0] + '.png'
            # Save image
            img.save(os.path.join(output_path, file_name))
        else:
            try:
                # Save image
                img.save(output_path)
            except OSError as e:
                raise OSError("Error! "
                              "Please indicate the correct extension of the final file, for example: .png",
                              "Error: ", e)
    else:
        # Change file extension to png
        file_name = os.path.splitext(file_name)[0] + '.png'
        # Save image
        img.save(os.path.join(output_path, file_name))


def cli():
    """CLI"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True,
                        help="Path to input file or dir.", action="store", dest="input_path")

    args = parser.parse_args()
    # Parse arguments
    input_path = args.input_path
    output_path = "output.png"
    if input_path is None:
        raise Exception("Please specify input path.")
    
    model = model_detect()  # Load model
    
    wmode = __work_mode__(input_path)  # Get work mode
    if wmode == "file":  # File work mode
        image = model.process_image(input_path)
        __save_image_file__(image, os.path.basename(input_path), output_path, wmode)
    elif wmode == "dir":  # Dir work mode
        # Start process
        files = os.listdir(input_path)
        for file in tqdm.tqdm(files, ascii=True, desc='Remove Background', unit='image'):
            file_path = os.path.join(input_path, file)
            image = model.process_image(file_path)
            __save_image_file__(image, file, output_path, wmode)
    else:
        raise Exception("Please indicate the correct path to the file or folder.")


if __name__ == "__main__":
    cli()
