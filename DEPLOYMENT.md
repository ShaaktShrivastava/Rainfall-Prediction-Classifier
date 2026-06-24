# Vercel Deployment Guide

## Prerequisites
- Vercel account (sign up at https://vercel.com)
- Vercel CLI installed (already installed on your system)

## Deployment Steps

### Option 1: Deploy via CLI (Recommended)

1. **Login to Vercel** (if not already logged in):
   ```bash
   vercel login
   ```

2. **Deploy to production**:
   ```bash
   vercel --prod
   ```

3. Follow the prompts:
   - Set up and deploy? **Y**
   - Which scope? Choose your account
   - Link to existing project? **N** (first time)
   - Project name? Press enter or provide a custom name
   - Directory? Press enter (current directory)
   - Override settings? **N**

### Option 2: Deploy via GitHub

1. Push your code to a GitHub repository
2. Go to https://vercel.com/new
3. Import your repository
4. Vercel will auto-detect the configuration
5. Click "Deploy"

## Configuration Files Created

- `vercel.json` - Vercel deployment configuration
- `.vercelignore` - Files to exclude from deployment
- Updated `requirements.txt` - Removed dev dependencies (matplotlib, seaborn, uvicorn)

## Important Notes

- **Model file size**: 12MB (within Vercel's 50MB limit)
- **Cold starts**: First request after inactivity may be slow
- **Serverless limits**: 10s execution timeout on Hobby plan, 60s on Pro
- The app runs in serverless mode, not as a persistent server

## Verify Deployment

After deployment, test these endpoints:
- `GET /` - Web interface
- `POST /predict` - Prediction API
- `GET /health` - Health check
- `GET /docs` - FastAPI interactive docs

## Troubleshooting

If deployment fails:
1. Check build logs in Vercel dashboard
2. Ensure model.pkl and feature_info.pkl are committed to git
3. Verify all dependencies are in requirements.txt
4. Check that Python version is compatible (Vercel uses Python 3.9+)

## Local Testing

Before deploying, test locally:
```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Visit http://localhost:8000 to verify everything works.
