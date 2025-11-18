from auth_face import FaceRecognitionApp, FaceRecognitionSettings
from auth_face.utils import load_image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

service = FaceRecognitionApp(settings=FaceRecognitionSettings())
api = FastAPI(title="FaceAuth API")


class ImagePayload(BaseModel):
    image: str


class AuthRequest(ImagePayload):
    issue_session: bool = False


class EnrollRequest(ImagePayload):
    name: str


class LogoutRequest(BaseModel):
    token: str


def _frame_from_payload(payload: ImagePayload):
    try:
        return load_image(payload.image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _serialize_detection(detection):
    return {
        "label": detection.label,
        "similarity": detection.similarity,
        "score": detection.score,
        "bbox": detection.bbox,
    }


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/auth")
def auth(payload: AuthRequest):
    frame = _frame_from_payload(payload)
    detection, token = service.authenticate(frame, issue_session=payload.issue_session)
    if not detection:
        raise HTTPException(status_code=404, detail="Face not detected")
    data = _serialize_detection(detection)
    data["token"] = token
    return data


@api.post("/enroll")
def enroll(payload: EnrollRequest):
    frame = _frame_from_payload(payload)
    try:
        detection = service.enroll(payload.name, frame)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _serialize_detection(detection)


@api.get("/faces")
def faces():
    return [
        {"id": face.id, "name": face.name, "created_at": face.created_at}
        for face in service.list_faces()
    ]


@api.post("/logout")
def logout(payload: LogoutRequest):
    if not service.logout(payload.token):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "revoked"}


@api.get("/sessions/{token}")
def session(token: str):
    session_data = service.validate_session(token)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"token": session_data.token, "name": session_data.name, "issued_at": session_data.issued_at}


@api.on_event("shutdown")
def shutdown():
    service.close()


def main() -> None:
    uvicorn.run("index:api", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()