# MTSF-Benchmarking

## To run the code:
```bash
docker build -t iiai-dev .
```
```bash
docker run -it --rm \
  --gpus all \
  --shm-size=8g \
  -v "$(pwd):/app" \
  iiai-dev
```