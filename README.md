This build is now a small face auth API. You send a base64 image, it detects with Mediapipe, embeds with SigLIP, matches against SQLite, and can enroll or reject. Sessions sit in memory so anything can ask for a token or clear it later.

Everything lives in the same Python package as before, just without the old camera window. FastAPI + uvicorn handle the HTTP part, configs stay in `FaceRecognitionSettings`, and the utils provide quick image loading and ROI prep. It’s basic but ready to drop into other projects.

