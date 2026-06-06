# AWS Dental API Instance Management

## Instance Details
- **Instance ID**: `i-0e282057a78446000`
- **Public IP**: `3.239.246.172` (changes when stopped/started)
- **Instance Type**: `g4dn.xlarge` (1x T4 GPU, 16GB VRAM)
- **Region**: `us-east-1`
- **AMI**: Align Official NVIDIA AMI - Amazon Linux 2023
- **Username**: `ec2-user`

## API Endpoints
- **Health**: http://3.239.246.172:8000/health
- **Analyze**: POST http://3.239.246.172:8000/analyze (multipart form: `file` + optional `question`)

## SSH / WinSCP Connection

Use the `dental-api-winscp.pem` key file (in this folder):

### SSH from Command Line
```powershell
ssh -i "c:\DEntal-api analyse\dental-api-winscp.pem" ec2-user@3.239.246.172
```

### WinSCP Setup
1. **Host**: `3.239.246.172` (or current IP after restart)
2. **Port**: `22`
3. **Username**: `ec2-user`
4. **Authentication**: Private key file → Browse to `dental-api-winscp.pem`
   - WinSCP will auto-convert to PPK format

### SCP Upload Example
```powershell
scp -i "c:\DEntal-api analyse\dental-api-winscp.pem" "c:\DEntal-api analyse\dental_api_aws.py" ec2-user@3.239.246.172:~/
```

## Quick Commands

### Check Instance Status
```powershell
aws ec2 describe-instances --instance-ids i-0e282057a78446000 --query "Reservations[0].Instances[0].[State.Name,PublicIpAddress]" --output text
```

### Start Instance
```powershell
aws ec2 start-instances --instance-ids i-0e282057a78446000
aws ec2 wait instance-running --instance-ids i-0e282057a78446000
aws ec2 describe-instances --instance-ids i-0e282057a78446000 --query "Reservations[0].Instances[0].PublicIpAddress" --output text
```

### Stop Instance (saves money!)
```powershell
aws ec2 stop-instances --instance-ids i-0e282057a78446000
```

### Check API Health
```powershell
curl.exe http://<PUBLIC_IP>:8000/health
```

### Test Analyze Endpoint
```powershell
curl.exe -X POST -F "file=@path/to/dental-xray.jpg" http://<PUBLIC_IP>:8000/analyze
```

## Server Details

### Disk Setup
- `/data` - 100GB EBS volume mounted for model storage
- `HF_HOME=/data/huggingface` - HuggingFace cache location

### Start API (on EC2)
```bash
cd ~
HF_HOME=/data/huggingface nohup python3 dental_api_aws.py > api.log 2>&1 &
```

### Check API Logs
```bash
tail -f ~/api.log
```

### Check GPU Usage
```bash
nvidia-smi
```

## Daily Workflow

### Start of Work Day
1. Start instance: `aws ec2 start-instances --instance-ids i-0e282057a78446000`
2. Wait ~2 min, get new IP
3. SSH in and run:
   ```bash
   HF_HOME=/data/huggingface nohup python3 ~/dental_api_aws.py > ~/api.log 2>&1 &
   ```
4. Wait ~3-5 min for model to load
5. Test: `curl http://<NEW_IP>:8000/health`

### End of Work Day
```powershell
aws ec2 stop-instances --instance-ids i-0e282057a78446000
```

## After 5 Days - Cleanup

### Terminate Instance (DELETE - cannot undo!)
```powershell
aws ec2 terminate-instances --instance-ids i-0e282057a78446000
```

### Delete All Resources
```powershell
# Delete key pair
aws ec2 delete-key-pair --key-name dental-api-key

# Delete security group (after instance terminated)
aws ec2 delete-security-group --group-id sg-0078fa6a04d2f11c0

# Detach and delete internet gateway
aws ec2 detach-internet-gateway --internet-gateway-id igw-0380ecf20365d5242 --vpc-id vpc-0c2134526659f2be8
aws ec2 delete-internet-gateway --internet-gateway-id igw-0380ecf20365d5242

# Delete subnet
aws ec2 delete-subnet --subnet-id subnet-00e82d413a123fb40

# Delete route table
aws ec2 delete-route-table --route-table-id rtb-0978350f88bfaf5b7

# Delete VPC
aws ec2 delete-vpc --vpc-id vpc-0c2134526659f2be8
```

## Cost Tracking
- g4dn.xlarge: ~$0.526/hour
- Storage (100GB gp3): ~$8/month
- Running 4 hours/day × 5 days = ~$10.50 total
