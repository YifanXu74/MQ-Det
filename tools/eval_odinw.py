import argparse
import os
from pathlib import Path

dataset_configs = {
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

# dataset_configs = {
#                     # "AerialMaritimeDrone_large": "configs/odinw_13/AerialMaritimeDrone_large.yaml",
#                     # "AerialMaritimeDrone_tiled": "configs/odinw_35/AerialMaritimeDrone_tiled.yaml",
#                     # "AmericanSignLanguageLetters": "configs/odinw_35/AmericanSignLanguageLetters_American_Sign_Language_Letters.v1-v1.coco.yaml",
#                     # "Aquarium": "configs/odinw_13/Aquarium_Aquarium_Combined.v2-raw-1024.coco.yaml",
#                     # "BCCD_BCCD": "configs/odinw_35/BCCD_BCCD.v3-raw.coco.yaml",
#                     # "ChessPiece": "configs/odinw_35/ChessPieces_Chess_Pieces.v23-raw.coco.yaml",
#                     # "CottontailRabbits": "configs/odinw_13/CottontailRabbits.yaml",
#                     # "DroneControl_Drone_Control": "configs/odinw_35/DroneControl_Drone_Control.v3-raw.coco.yaml",
#                     # "EgoHands_generic": "configs/odinw_13/EgoHands_generic.yaml",
#                     # "EgoHands_speciﬁc": "configs/odinw_35/EgoHands_specific.yaml",
#                     # "HardHatWorkers": "configs/odinw_35/HardHatWorkers_raw.yaml",
#                     # "MaskWearing": "configs/odinw_35/MaskWearing_raw.yaml",
#                     # "MountainDewCommercial": "configs/odinw_35/MountainDewCommercial.yaml",
#                     # "NorthAmericaMushrooms": "configs/odinw_13/NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco.yaml",
#                     # "OxfordPets_by-breed": "configs/odinw_35/OxfordPets_by-breed.yaml",
#                     # "OxfordPets_by-species": "configs/odinw_35/OxfordPets_by-species.yaml",
#                     # "PKLot_640": "configs/odinw_35/PKLot_640.yaml",
#                     # "Packages": "configs/odinw_13/Packages_Raw.yaml",
#                     # "Raccoon_Raccoon": "configs/odinw_13/Raccoon_Raccoon.v2-raw.coco.yaml",
#                     # "ShellﬁshOpenImages": "configs/odinw_13/ShellfishOpenImages_raw.yaml",
#                     # "ThermalCheetah": "configs/odinw_35/ThermalCheetah.yaml",
#                     # "UnoCards": "configs/odinw_35/UnoCards_raw.yaml",
#                     # "VehiclesOpenImages": "configs/odinw_13/VehiclesOpenImages_416x416.yaml",
#                     # "WildﬁreSmoke": "configs/odinw_35/WildfireSmoke.yaml",
#                     # "boggleBoards": "configs/odinw_35/boggleBoards_416x416AutoOrient_export_.yaml",
#                     # "brackishUnderwater": "configs/odinw_35/brackishUnderwater_960x540.yaml",
#                     # "dice_mediumColor": "configs/odinw_35/dice_mediumColor_export.yaml",
#                     # "openPoetryVision": "configs/odinw_35/openPoetryVision_512x512.yaml",
#                     # "pistols": "configs/odinw_13/pistols_export.yaml",
#                     # "plantdoc": "configs/odinw_35/plantdoc_416x416.yaml",
#                     # "pothole": "configs/odinw_13/pothole.yaml",
#                     "selfdrivingCar": "configs/odinw_35/selfdrivingCar_fixedLarge_export_.yaml",
#                     "thermalDogsAndPeople": "configs/odinw_13/thermalDogsAndPeople.yaml",
#                     "websiteScreenshots": "configs/odinw_35/websiteScreenshots.yaml",
#                     "PascalVOC": "configs/odinw_13/PascalVOC.yaml",
#                 }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Finetuning")
    parser.add_argument("--python", default='python3.9',type=str)
    parser.add_argument(
            "--config_file",
            default="",
            metavar="FILE",
            help="path to config file",
            type=str,
        )
    parser.add_argument("--opts", default="", type=str)
    parser.add_argument("--setting", default="zero-shot", type=str)
    parser.add_argument("--add_name", default="", type=str)
    parser.add_argument("--log_path", default="odinw_log", type=str)
    
    args = parser.parse_args()
    
    os.makedirs(args.log_path, exist_ok=True)

    if args.setting == "zero-shot":
        for dataset_name ,task_config in dataset_configs.items():
            query_bank_path = 'MODEL/{}_query_5_pool7_sel_{}.pth'.format(dataset_name, args.add_name)
            log_save_path = str(Path(args.log_path, '{}-{}-zero-shot.txt'.format(dataset_name, args.add_name)))

            cmd = '{} tools/test_grounding_net.py --config-file {} --task_config {} --additional_model_config configs/vision_query_5shot/odinw.yaml VISION_QUERY.NUM_QUERY_PER_CLASS 100 VISION_QUERY.QUERY_BANK_PATH {} TEST.IMS_PER_BATCH 1 {} > {}'\
                .format(
                    args.python,
                    args.config_file,
                    task_config,
                    query_bank_path,
                    args.opts,
                    log_save_path
                        )
            os.system(cmd)
        


