from pathlib import Path
from typing import List
import csv
from metrics.clustering_metrics import (
    exact_match,
    partial_match,
)


def read_images_from_directory(
    train_dir_path: str,
    test_dir_path: str,
    image_extensions: set,
) -> tuple[list[Path], list[Path]]:
    train_images = sorted(
        [
            f
            for f in Path(train_dir_path).iterdir()
            if f.suffix.lower() in image_extensions
        ]
    )
    print(f"Найдено {len(train_images)} изображений в папке `{train_dir_path}`")

    test_images = sorted(
        [
            f
            for f in Path(test_dir_path).iterdir()
            if f.suffix.lower() in image_extensions
        ]
    )
    print(f"Найдено {len(test_images)} изображений в папке `{test_dir_path}`")

    return train_images, test_images


def save_paths_to_file(file_path: Path, pred_paths: List[Path] | List[List], format: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "txt":
        with open(file_path, "w", encoding="utf-8") as file:
            for path in pred_paths:
                file.write(str(path.name) + "\n")
            print(f"Список изображений успешно сохранен в файл: {file_path}")

    if format == "csv":
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Image', 'IsLeaked'])

            for idx in range(len(pred_paths[0])):
                writer.writerow([pred_paths[0][idx], pred_paths[1][idx]])
            print(f"Ответы успешно сохранены в файл: {file_path}")

    
def print_save_results_clustering(
    true_groups: list[list[str]],
    preds_groups: list[list[str]],
    output_dir_path: Path,
    output_file_name: str,
) -> None:

    if true_groups:
        print(f"Точное совпадение групп: {exact_match(true_groups, preds_groups)}")
        print(f"Частичное совпадение групп: {partial_match(true_groups, preds_groups)}")
    with open(output_dir_path / output_file_name, "w") as f:
        for group in preds_groups:
            f.write(f"{','.join(group)}\n")