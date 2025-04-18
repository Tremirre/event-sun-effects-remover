import pathlib

SRC_PATH = pathlib.Path("data") / "processed"
SRC_GLOB = "**/*.npy"
TARGET_PATH = pathlib.Path("data") / "inpaint"

ASSINGMENT = {
    "train": [
        "001-town01",
        "001-town02",
        "interlaken_00_c",
        "interlaken_00_f",
        "zurich_00_b",
        "zurich_city_05_b",
        "zurich_city_06_a",
        "back3",
        "back4",
        "highway2",
        "cityscapes_all",
    ],
    "val": [
        "interlaken_00_g",
        "zurich_city_07_a",
        "back1",
        "back9",
        "street2",
    ],
    "test": [
        "zurich_01_a",
        "zurich_city_13_b",
        "highway1",
    ],
}

if __name__ == "__main__":
    for split, scenes in ASSINGMENT.items():
        split_path = TARGET_PATH / split
        split_path.mkdir(parents=True, exist_ok=True)

        for scene in scenes:
            scene_path = SRC_PATH / scene
            if not scene_path.exists():
                print(f"Scene {scene} does not exist.")
                continue
            for file in scene_path.glob(SRC_GLOB):
                if not file.is_file():
                    print(f"File {file} is not a file.")
                    continue
                target_path = split_path / file.relative_to(scene_path)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(file.read_text())
                print(f"Copied {file} to {target_path}")
