from pathlib import Path
from metrics.compare_images_metrics import compute_metrics
from utils.read_save_files import read_images_from_directory, save_paths_to_file
import scripts.detect_with_hash_methods as hm
import scripts.detect_with_ssim_method as ssm
from scripts.detect_with_cnn import TransformerEmbedder
from utils.log import TeeLoggerContext
from utils.measure_time import measure_time
import time

def open_files(
    train_images_path, test_images_path, image_extensions, leakage_images_path
):
    train_images, test_images = read_images_from_directory(
        train_images_path, test_images_path, image_extensions
    )

    with open(leakage_images_path, "r", encoding="utf-8") as file:
        leakage_images = file.read().split("\n")
    print(
        f"Количество лишних изображений в оригинальной разметке: {len(leakage_images)}"
    )

    return train_images, test_images, leakage_images


if __name__ == "__main__":

    log_path = Path("logs/predict_on_public_data.log")
    log_path.parent.mkdir(exist_ok=True)

    with TeeLoggerContext(log_path):

        print("Public data")
        input_dir_path = Path("datasets\public_data")
        train_images_path = f"{input_dir_path}/train"
        test_images_path = f"{input_dir_path}/test"
        leakage_images_path = f"{input_dir_path}/leakage_files.txt"
        output_dir_path = "output"
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        
        print("Метод dHash (Difference Hash)")
        start_time = time.time()
        train_images, test_images, leakage_images = open_files(
            train_images_path, test_images_path, image_extensions, leakage_images_path
        )
        output_path = Path(f"{output_dir_path}/pred_cached_dHash.txt")
        preds = hm.get_image_comparison(
            train_images, test_images, threshold=15, comparison_method="dhash"
        )
        end_time = time.time()
        dhash_preds_time = measure_time(start_time, end_time)
        metrics_dhash = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        
        start_time = time.time()
        train_images, test_images, leakage_images = open_files(
            train_images_path, test_images_path, image_extensions, leakage_images_path
        )
        output_path = Path(f"{output_dir_path}/pred_ssim.txt")

        IMAGE_SIZE = (64, 64)  

        print("Загрузка тренировочных изображений...")
        train_images_data = ssm.load_images(train_images, IMAGE_SIZE)

        print("Загрузка тестовых изображений...")
        test_images_data = ssm.load_images(test_images, IMAGE_SIZE)

        preds = ssm.get_image_comparison(
            train_images=train_images_data,
            test_images=test_images_data,
            test_paths=test_images,
            threshold=0.2,  
        )
        end_time = time.time()
        ssim_preds_time = measure_time(start_time, end_time)

        metrics_ssim = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        
        print("ResNet50")
        start_time = time.time()
        output_path = Path(f"{output_dir_path}/pred_resnet50.txt")
        train_images, test_images, leakage_images = open_files(
            train_images_path, test_images_path, image_extensions, leakage_images_path
        )
        IMAGE_SIZE = (224, 224)
        embedder = TransformerEmbedder(IMAGE_SIZE, model_name="resnet50", device="cuda")

        print("Создание эмбеддингов для тренировочных изображений...")
        train_embeddings = embedder.compute_embeddings(train_images)

        print("Создание эмбеддингов для тестовых изображений...")
        test_embeddings = embedder.compute_embeddings(test_images)

        preds = embedder.get_image_comparison(
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            test_paths=test_images,
            threshold=0.75,
        )
        end_time = time.time()
        resnet_preds_time = measure_time(start_time, end_time)

        metrics_resnet = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        resnet_preds = preds