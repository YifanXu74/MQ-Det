import argparse
import os

odinw13_configs = {
                'AerialDrone': 'configs/odinw_13/AerialMaritimeDrone_large.yaml',
                'Aquarium': 'configs/odinw_13/Aquarium_Aquarium_Combined.v2-raw-1024.coco.yaml',
                'Rabbits': 'configs/odinw_13/CottontailRabbits.yaml',
                'EgoHands': 'configs/odinw_13/EgoHands_generic.yaml',
                'Mushrooms': 'configs/odinw_13/NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco.yaml',
                'Packages': 'configs/odinw_13/Packages_Raw.yaml',
                'Pistols':'configs/odinw_13/pistols_export.yaml',
                'Pothole': 'configs/odinw_13/pothole.yaml ',
                'Raccoon': 'configs/odinw_13/Raccoon_Raccoon.v2-raw.coco.yaml',
                'Shellﬁsh': 'configs/odinw_13/ShellfishOpenImages_raw.yaml',
                'Thermal' : 'configs/odinw_13/thermalDogsAndPeople.yaml',
                'Vehicles': 'configs/odinw_13/VehiclesOpenImages_416x416.yaml',
                'PascalVOC': 'configs/odinw_13/PascalVOC.yaml',
                }

odinw35_configs = {
                "AerialMaritimeDrone_large": "configs/odinw_13/AerialMaritimeDrone_large.yaml",
                "AerialMaritimeDrone_tiled": "configs/odinw_35/AerialMaritimeDrone_tiled.yaml",
                "AmericanSignLanguageLetters": "configs/odinw_35/AmericanSignLanguageLetters_American_Sign_Language_Letters.v1-v1.coco.yaml",
                "Aquarium": "configs/odinw_13/Aquarium_Aquarium_Combined.v2-raw-1024.coco.yaml",
                "BCCD_BCCD": "configs/odinw_35/BCCD_BCCD.v3-raw.coco.yaml",
                "ChessPiece": "configs/odinw_35/ChessPieces_Chess_Pieces.v23-raw.coco.yaml",
                "CottontailRabbits": "configs/odinw_13/CottontailRabbits.yaml",
                "DroneControl_Drone_Control": "configs/odinw_35/DroneControl_Drone_Control.v3-raw.coco.yaml",
                "EgoHands_generic": "configs/odinw_13/EgoHands_generic.yaml",
                "EgoHands_speciﬁc": "configs/odinw_35/EgoHands_specific.yaml",
                "HardHatWorkers": "configs/odinw_35/HardHatWorkers_raw.yaml",
                "MaskWearing": "configs/odinw_35/MaskWearing_raw.yaml",
                "MountainDewCommercial": "configs/odinw_35/MountainDewCommercial.yaml",
                "NorthAmericaMushrooms": "configs/odinw_13/NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco.yaml",
                "OxfordPets_by-breed": "configs/odinw_35/OxfordPets_by-breed.yaml",
                "OxfordPets_by-species": "configs/odinw_35/OxfordPets_by-species.yaml",
                "PKLot_640": "configs/odinw_35/PKLot_640.yaml",
                "Packages": "configs/odinw_13/Packages_Raw.yaml",
                "Raccoon_Raccoon": "configs/odinw_13/Raccoon_Raccoon.v2-raw.coco.yaml",
                "ShellﬁshOpenImages": "configs/odinw_13/ShellfishOpenImages_raw.yaml",
                "ThermalCheetah": "configs/odinw_35/ThermalCheetah.yaml",
                "UnoCards": "configs/odinw_35/UnoCards_raw.yaml",
                "VehiclesOpenImages": "configs/odinw_13/VehiclesOpenImages_416x416.yaml",
                "WildﬁreSmoke": "configs/odinw_35/WildfireSmoke.yaml",
                "boggleBoards": "configs/odinw_35/boggleBoards_416x416AutoOrient_export_.yaml",
                "brackishUnderwater": "configs/odinw_35/brackishUnderwater_960x540.yaml",
                "dice_mediumColor": "configs/odinw_35/dice_mediumColor_export.yaml",
                "openPoetryVision": "configs/odinw_35/openPoetryVision_512x512.yaml",
                "pistols": "configs/odinw_13/pistols_export.yaml",
                "plantdoc": "configs/odinw_35/plantdoc_416x416.yaml",
                "pothole": "configs/odinw_13/pothole.yaml",
                "selfdrivingCar": "configs/odinw_35/selfdrivingCar_fixedLarge_export_.yaml",
                "thermalDogsAndPeople": "configs/odinw_13/thermalDogsAndPeople.yaml",
                "websiteScreenshots": "configs/odinw_35/websiteScreenshots.yaml",
                "PascalVOC": "configs/odinw_13/PascalVOC.yaml",
                }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Finetuning")
    parser.add_argument("--python", default='python',type=str, help='sometime need to assign a new python command, e.g., python3, python3.9...')
    parser.add_argument(
            "--config_file",
            default="",
            metavar="FILE",
            help="path to config file",
            type=str,
        )
    parser.add_argument("--dataset", default="", type=str)
    parser.add_argument("--num_vision_queries", default=5, type=int)
    parser.add_argument("--add_name", default="", type=str)
    args = parser.parse_args()

    if args.dataset == 'objects365':
        # extract vision queries for modulated pre-training, num_vision_queries=5000 for default
        save_path = 'MODEL/object365_query_5000_sel_{}.pth'.format(args.add_name)
        cmd = '{} tools/train_net.py --config-file {} --extract_query VISION_QUERY.QUERY_BANK_SAVE_PATH {} VISION_QUERY.QUERY_BANK_PATH ""'\
                .format(
                        args.python,
                        args.config_file,
                        save_path,
                        )
        os.system(cmd)
    elif args.dataset == 'lvis':
        save_path = 'MODEL/lvis_query_{}_pool7_sel_{}.pth'.format(args.num_vision_queries, args.add_name)
        cmd = '{} tools/train_net.py --config-file {} --additional_model_config configs/vision_query_5shot/lvis_minival.yaml --extract_query VISION_QUERY.QUERY_BANK_SAVE_PATH {} VISION_QUERY.MAX_QUERY_NUMBER {} DATASETS.FEW_SHOT {} VISION_QUERY.QUERY_BANK_PATH ""'\
                .format(
                        args.python,
                        args.config_file,
                        save_path,
                        args.num_vision_queries, args.num_vision_queries,
                        )
        os.system(cmd)
    elif args.dataset == 'odinw-13':
        for name, config in odinw13_configs.items():
            save_path = 'MODEL/{}_query_{}_pool7_sel_{}.pth'.format(name, args.num_vision_queries, args.add_name)
            cmd = '{} tools/train_net.py --config-file {} --additional_model_config configs/vision_query_5shot/odinw.yaml --task_config {} --extract_query VISION_QUERY.QUERY_BANK_SAVE_PATH {} VISION_QUERY.MAX_QUERY_NUMBER {} DATASETS.FEW_SHOT {} VISION_QUERY.DATASET_NAME {} VISION_QUERY.QUERY_BANK_PATH ""'\
                    .format(
                            args.python,
                            args.config_file,
                            config,
                            save_path,
                            args.num_vision_queries, args.num_vision_queries,
                            name,
                            )
            os.system(cmd)
    elif args.dataset == 'odinw-35':
        for name, config in odinw35_configs.items():
            save_path = 'MODEL/{}_query_{}_pool7_sel_{}.pth'.format(name, args.num_vision_queries, args.add_name)
            cmd = '{} tools/train_net.py --config-file {} --additional_model_config configs/vision_query_5shot/odinw.yaml --task_config {} --extract_query VISION_QUERY.QUERY_BANK_SAVE_PATH {} VISION_QUERY.MAX_QUERY_NUMBER {} DATASETS.FEW_SHOT {} VISION_QUERY.DATASET_NAME {} VISION_QUERY.QUERY_BANK_PATH ""'\
                    .format(
                            args.python,
                            args.config_file,
                            config,
                            save_path,
                            args.num_vision_queries, args.num_vision_queries,
                            name,
                            )
            os.system(cmd)