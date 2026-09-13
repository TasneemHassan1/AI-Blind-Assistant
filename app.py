"""Executable entry point for the AI Blind Assistant."""

from camera import Camera
from controller import AssistantController
from voice import speak
from voice_commands import VoiceCommandListener


def main() -> None:
    controller = AssistantController(speak=speak)
    commands = VoiceCommandListener(
        on_wake=controller.begin_question,
        on_question=controller.handle_question,
        on_cancel=controller.cancel_question,
    )
    commands.start()
    try:
        Camera(controller).run()
    finally:
        commands.stop()
        controller.shutdown()


if __name__ == "__main__":
    main()