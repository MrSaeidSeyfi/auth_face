from auth_face import FaceRecognitionApp, FaceRecognitionSettings


def main() -> None:
    app = FaceRecognitionApp(settings=FaceRecognitionSettings())
    app.run()


if __name__ == "__main__":
    main()