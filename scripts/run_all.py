import subprocess
import os
import sys

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts_dir = os.path.join(base_dir, "scripts")
    
    scripts = [
        "train_mlp.py",
        "prune.py",
        "encode_csr.py",
        "verify.py",
        "benchmark_py.py"
    ]
    
    for script in scripts:
        print(f"\n{'='*50}\nRunning {script}...\n{'='*50}")
        script_path = os.path.join(scripts_dir, script)
        try:
            subprocess.run([sys.executable, script_path], check=True, cwd=base_dir)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")
            sys.exit(1)
            
    print("\nAll steps completed successfully!")

if __name__ == "__main__":
    main()
