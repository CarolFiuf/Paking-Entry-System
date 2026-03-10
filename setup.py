#!/usr/bin/env python3
"""
Setup: verify GPU, setup PostgreSQL + pgvector, download models.
  python setup.py              # Full setup
  python setup.py --check      # Chỉ kiểm tra
  python setup.py --init-db    # Chỉ setup database
"""
import subprocess, os, argparse, sys


def check_gpu():
    print("\n[1/6] GPU")
    found = False

    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"  ✅ PyTorch CUDA: {name} ({mem:.1f}GB)")
            found = True
        else:
            print("  ⚠️  PyTorch CUDA not available "
                  "(không ảnh hưởng nếu dùng onnxruntime)")
    except ImportError:
        print("  ⚠️  PyTorch chưa cài")

    # Check ONNX Runtime CUDA (quan trọng hơn cho insightface)
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            print(f"  ✅ ONNX Runtime CUDA available")
            found = True
        if "TensorrtExecutionProvider" in providers:
            print(f"  ✅ ONNX Runtime TensorRT available")
    except ImportError:
        pass

    if not found:
        print("  ❌ Không tìm thấy GPU provider nào!")
    return found


def check_tensorrt():
    print("\n[2/6] TensorRT")
    try:
        import tensorrt
        print(f"  ✅ TensorRT {tensorrt.__version__}")
    except ImportError:
        print("  ⚠️  Chưa cài (cần JetPack SDK)")


def setup_postgres():
    print("\n[3/6] PostgreSQL + pgvector")

    # Check psql available
    try:
        r = subprocess.run(["psql", "--version"], capture_output=True, text=True)
        print(f"  ✅ {r.stdout.strip()}")
    except FileNotFoundError:
        print("  ❌ PostgreSQL chưa cài!")
        print("     sudo apt install postgresql postgresql-contrib")
        print("     sudo apt install postgresql-16-pgvector")
        return False

    # Check pgvector extension
    try:
        import psycopg2
        print(f"  ✅ psycopg2 {psycopg2.__version__}")
    except ImportError:
        print("  ❌ psycopg2 chưa cài: pip install psycopg2-binary")
        return False

    try:
        import pgvector
        print("  ✅ pgvector Python binding")
    except ImportError:
        print("  ❌ pgvector chưa cài: pip install pgvector")
        return False

    print()
    print("  Tạo database (cần quyền postgres):")
    print("    sudo -u postgres createuser parking --pwprompt")
    print("    sudo -u postgres createdb parking --owner=parking")
    print("    sudo -u postgres psql -d parking -c 'CREATE EXTENSION vector;'")
    print()
    print("  Hoặc chạy lệnh tự động:")
    print("    python setup.py --init-db")

    return True


def init_db():
    """Tạo user, database, extension."""
    print("\n  Tạo PostgreSQL database...")

    cmds = [
        ("Tạo user", "sudo -u postgres psql -c "
         "\"CREATE USER parking WITH PASSWORD 'parking123';\""),
        ("Tạo database", "sudo -u postgres psql -c "
         "\"CREATE DATABASE parking OWNER parking;\""),
        ("Cài pgvector extension", "sudo -u postgres psql -d parking -c "
         "\"CREATE EXTENSION IF NOT EXISTS vector;\""),
        ("Grant quyền", "sudo -u postgres psql -d parking -c "
         "\"GRANT ALL PRIVILEGES ON DATABASE parking TO parking;\""),
    ]

    for name, cmd in cmds:
        print(f"  {name}...", end=" ")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode == 0:
            print("✅")
        else:
            err = r.stderr.strip()
            if "already exists" in err:
                print("⏭️  (đã có)")
            else:
                print(f"❌ {err}")

    # Test connection
    print("\n  Test connection...", end=" ")
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost", port=5432,
            dbname="parking", user="parking", password="parking123")
        cur = conn.cursor()
        cur.execute("SELECT 1")
        conn.close()
        print("✅ Connected!")
    except Exception as e:
        print(f"❌ {e}")


def setup_plate_model():
    print("\n[4/6] Plate Detection Model")
    model_path = "./models/plate_yolov8n.pt"
    if os.path.exists(model_path):
        print(f"  ✅ Đã có: {model_path}")
        return

    print("  📥 Download YOLOv8n base model...")
    from ultralytics import YOLO
    YOLO("yolov8n.pt")
    os.makedirs("./models", exist_ok=True)
    os.rename("yolov8n.pt", model_path)
    print(f"  ✅ Saved: {model_path}")
    print("  ⚠️  Fine-tune trên biển số VN để tăng accuracy:")
    print("     yolo train model=yolov8n.pt data=plates.yaml epochs=100")


def setup_face_model():
    print("\n[5/6] Face Model (InsightFace)")
    try:
        from insightface.app import FaceAnalysis
        import numpy as np
        app = FaceAnalysis(
            name="buffalo_sc",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        app.get(dummy)
        print("  ✅ buffalo_sc ready")
    except Exception as e:
        print(f"  ❌ {e}")


def setup_ocr():
    print("\n[6/6] PaddleOCR")
    try:
        from paddleocr import PaddleOCR
        import numpy as np
        try:
            ocr = PaddleOCR(lang="en", use_angle_cls=False,
                            show_log=False, use_gpu=True)
        except TypeError:
            ocr = PaddleOCR(lang="en", use_textline_orientation=False,
                            use_gpu=True)
        dummy = np.random.randint(0, 255, (32, 160, 3), dtype=np.uint8)
        ocr.ocr(dummy, cls=False)
        print("  ✅ PaddleOCR ready (GPU)")
    except Exception as e:
        print(f"  ❌ {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--init-db", action="store_true")
    args = parser.parse_args()

    print("=" * 50)
    print("  Smart Parking System — Setup")
    print("=" * 50)

    if args.init_db:
        init_db()
        return

    check_gpu()
    check_tensorrt()
    setup_postgres()

    if not args.check:
        setup_plate_model()
        setup_face_model()
        setup_ocr()

    print("\n" + "=" * 50)
    print("  Done! Next:")
    print("    python setup.py --init-db      # Tạo database")
    print("    python main.py --entry test.mp4 # Test")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
