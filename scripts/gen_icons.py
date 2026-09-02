"""Generate placeholder icons for Aura Studio."""
import struct, zlib, os

ICON_DIR = os.path.join(os.path.dirname(__file__), "..", "src-tauri", "icons")


def create_png(width, height, filename, r=37, g=99, b=235):
    raw_data = b""
    for y in range(height):
        raw_data += b"\x00"
        for x in range(width):
            raw_data += struct.pack("BBBB", r, g, b, 255)

    def chunk(chunk_type, data):
        c = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    idat = zlib.compress(raw_data)

    with open(filename, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", idat))
        f.write(chunk(b"IEND", b""))


def create_ico(filename, sizes=None):
    if sizes is None:
        sizes = [(32, 32), (16, 16), (48, 48), (256, 256)]

    def create_png_data(w, h, r=37, g=99, b=235):
        raw = b""
        for y in range(h):
            raw += b"\x00"
            for x in range(w):
                raw += struct.pack("BBBB", r, g, b, 255)

        def chunk(t, d):
            c = t + d
            return (
                struct.pack(">I", len(d))
                + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            )

        ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )

    png_data = [create_png_data(w, h) for w, h in sizes]

    with open(filename, "wb") as f:
        f.write(struct.pack("<HHH", 0, 1, len(sizes)))
        offset = 6 + 16 * len(sizes)
        for (w, h), png in zip(sizes, png_data):
            iw = 0 if w == 256 else w
            ih = 0 if h == 256 else h
            f.write(struct.pack("<BBBBHHII", iw, ih, 0, 0, 1, 32, len(png), offset))
            offset += len(png)
        for png in png_data:
            f.write(png)


def create_icns(filename):
    """Create a minimal ICNS file (macOS icon)."""
    # Just use PNG data in an ICNS wrapper
    import subprocess

    # Create a 128x128 PNG and wrap it in basic ICNS
    png_path = filename.replace(".icns", "_tmp.png")
    create_png(128, 128, png_path)
    
    with open(png_path, "rb") as f:
        png_data = f.read()
    
    os.remove(png_path)
    
    # ICNS: header + one icon entry (ic07 = 128x128 PNG)
    icon_type = b"ic07"
    entry_size = 8 + len(png_data)
    total_size = 8 + entry_size
    
    with open(filename, "wb") as f:
        f.write(b"icns")
        f.write(struct.pack(">I", total_size))
        f.write(icon_type)
        f.write(struct.pack(">I", entry_size))
        f.write(png_data)


if __name__ == "__main__":
    os.makedirs(ICON_DIR, exist_ok=True)
    
    create_png(32, 32, os.path.join(ICON_DIR, "32x32.png"))
    create_png(128, 128, os.path.join(ICON_DIR, "128x128.png"))
    create_png(256, 256, os.path.join(ICON_DIR, "128x128@2x.png"))
    create_ico(os.path.join(ICON_DIR, "icon.ico"))
    create_icns(os.path.join(ICON_DIR, "icon.icns"))
    
    print("All icons generated successfully!")
