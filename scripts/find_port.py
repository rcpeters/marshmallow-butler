"""Find the USB serial port for an SO-101 arm by unplugging it interactively."""

from __future__ import annotations

from serial.tools import list_ports


def _current_ports() -> set[str]:
    return {p.device for p in list_ports.comports()}


def main() -> None:
    before = _current_ports()
    print("Detected serial ports:", sorted(before) or ["<none>"])
    input("\nUnplug ONLY the SO-101 USB cable you want to identify, then press Enter...")
    after = _current_ports()

    removed = sorted(before - after)
    if not removed:
        print("No port disappeared. Make sure the USB cable was unplugged and try again.")
        return

    print(f"Found SO-101 port: {removed[0]}")
    if len(removed) > 1:
        print(f"Note: multiple ports disappeared: {removed}")
    input("Plug the USB cable back in, then press Enter to finish...")


if __name__ == "__main__":
    main()
