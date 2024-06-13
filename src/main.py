"""
    author: SPDKH
    date: Nov 2, 2023
"""
from src.utils import consts
from src.data.googlemap import GoogleMap


def main():
    args = consts.ARGS
    aerial_data = GoogleMap(
        args=args,
        map_type='satellite',
        data_dir=args.data_dir,
        overlap=consts.OVERLAP
        )
    aerial_data.config()

    # Prepar the dataset for a keras DNN task
    # augmented_data = aerial_data.config_dnn()


if __name__ == '__main__':
    main()
