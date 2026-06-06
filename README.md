# DIAL - Dental Image Analysis API

AI-powered dental X-ray analysis using **Qwen3-VL-8B** with a fine-tuned LoRA adapter.

## 🚀 Live API

- **Swagger UI**: http://3.239.246.172:8000/docs
- **Health Check**: http://3.239.246.172:8000/health
- **Analyze Endpoint**: POST http://3.239.246.172:8000/analyze

> ⚠️ IP changes when instance restarts. Check `aws_commands.md` for current status.

## 📋 API Usage

### Health Check
```bash
curl http://3.239.246.172:8000/health
```

### Analyze Dental X-ray
```bash
curl -X POST -F "file=@dental-xray.jpg" http://3.239.246.172:8000/analyze
```

### With Custom Question
```bash
curl -X POST \
  -F "file=@dental-xray.jpg" \
  -F "question=Is there any cavity in this image?" \
  http://3.239.246.172:8000/analyze
```

## 🔧 Files

| File | Description |
|------|-------------|
| `dental_api_aws.py` | Main API code (runs on EC2) |
| `dental_api_final.py` | Original Kaggle version (2x T4 GPUs) |
| `setup_server.sh` | EC2 setup script |
| `aws_commands.md` | AWS instance management commands |

## 🖥️ EC2 Instance

- **Type**: g4dn.xlarge (1x Tesla T4, 16GB VRAM)
- **Instance ID**: `i-0e282057a78446000`
- **Region**: us-east-1

See `aws_commands.md` for start/stop/SSH instructions.

## 🔑 SSH/WinSCP Access

Get the `dental-api-winscp.pem` key file from the team (not in repo for security).

```bash
ssh -i dental-api-winscp.pem ec2-user@<CURRENT_IP>
```

## 📊 Model Details

- **Base**: Qwen/Qwen3-VL-8B-Instruct
- **Adapter**: hrsvrn/Qwen3-VL-8B-dentex-rlvr-grpo
- **Quantization**: 4-bit (bitsandbytes NF4)
- **VRAM Usage**: ~8GB

## ⚠️ Disclaimer

This API is for **educational purposes only**. Not intended for clinical diagnosis.
