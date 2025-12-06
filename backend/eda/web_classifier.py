import http.server
import shutil
import socketserver
import urllib.parse
from pathlib import Path

import cv2

PORT = 8000
BASE_DIR = Path(__file__).parent.parent / "data" / "czyste"
CONVERTIVE_DIR = BASE_DIR / ".convertive"
TRUCK_DIR = BASE_DIR / ".truck"

CONVERTIVE_DIR.mkdir(exist_ok=True)
TRUCK_DIR.mkdir(exist_ok=True)


def get_next_folder():
    # List folders that are not hidden and not processed
    folders = []
    if not BASE_DIR.exists():
        return None

    for item in BASE_DIR.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            folders.append(item)
    folders.sort(key=lambda x: x.name)
    return folders[0] if folders else None


class ClassifierHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            folder = get_next_folder()
            if not folder:
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h1>All done! No more folders to classify.</h1>")
                return

            # Find image
            bmp_files = list(folder.glob("*czarno.bmp"))
            if not bmp_files:
                bmp_files = list(folder.glob("*.bmp"))

            img_name = bmp_files[0].name if bmp_files else None

            if not img_name:
                img_src = ""
                display_msg = "No image found"
            else:
                # URL encode the image name to handle spaces
                encoded_img_name = urllib.parse.quote(img_name)
                img_src = f"/image/{folder.name}/{encoded_img_name}"
                display_msg = ""

            html = f"""
            <html>
            <head>
                <title>Data Classifier</title>
                <style>
                    body {{ font-family: sans-serif; text-align: center; padding: 20px; background-color: #f0f0f0; }}
                    .container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: inline-block; }}
                    img {{ max-width: 800px; max-height: 600px; border: 1px solid #ccc; margin: 20px 0; display: block; }}
                    .btn {{ padding: 15px 30px; font-size: 18px; margin: 10px; cursor: pointer; border-radius: 4px; font-weight: bold; }}
                    .btn-c {{ background-color: #4CAF50; color: white; border: none; }}
                    .btn-c:hover {{ background-color: #45a049; }}
                    .btn-t {{ background-color: #f44336; color: white; border: none; }}
                    .btn-t:hover {{ background-color: #da190b; }}
                    .info {{ color: #666; margin-bottom: 10px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Classifying: {folder.name}</h2>
                    <div class="info">Press 'C' for Convertible, 'T' for Truck</div>
                    {f"<div>{display_msg}</div>" if display_msg else ""}
                    {f'<img src="{img_src}" />' if img_src else ""}
                    <div>
                        <form action="/move" method="post" style="display:inline">
                            <input type="hidden" name="folder" value="{folder.name}">
                            <input type="hidden" name="target" value="convertible">
                            <button type="submit" class="btn btn-c">Convertible (C)</button>
                        </form>
                        <form action="/move" method="post" style="display:inline">
                            <input type="hidden" name="folder" value="{folder.name}">
                            <input type="hidden" name="target" value="truck">
                            <button type="submit" class="btn btn-t">Truck (T)</button>
                        </form>
                    </div>
                </div>
                <script>
                    document.addEventListener('keydown', function(event) {{
                        if (event.key === 'c' || event.key === 'C') {{
                            document.querySelector('.btn-c').click();
                        }} else if (event.key === 't' || event.key === 'T') {{
                            document.querySelector('.btn-t').click();
                        }}
                    }});
                </script>
            </body>
            </html>
            """
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())

        elif self.path.startswith("/image/"):
            # Extract folder and image name
            # /image/folder_name/image_name
            try:
                # Unquote the path to handle spaces and special chars
                decoded_path = urllib.parse.unquote(self.path)
                parts = decoded_path.split("/")

                if len(parts) >= 4:
                    folder_name = parts[2]
                    image_name = parts[3]
                    # Security check: ensure no path traversal
                    if ".." in folder_name or ".." in image_name:
                        raise ValueError("Invalid path")

                    file_path = BASE_DIR / folder_name / image_name

                    if file_path.exists():
                        # Read image using cv2 and encode as JPEG
                        img = cv2.imread(str(file_path))
                        if img is not None:
                            ret, jpeg = cv2.imencode(".jpg", img)
                            if ret:
                                self.send_response(200)
                                self.send_header("Content-type", "image/jpeg")
                                self.end_headers()
                                self.wfile.write(jpeg.tobytes())
                            else:
                                self.send_error(500, "Could not encode image")
                        else:
                            self.send_error(500, "Could not read image")
                    else:
                        print(f"File not found: {file_path}")
                        self.send_error(404, "File not found")
                else:
                    self.send_error(400, "Bad request")
            except Exception as e:
                print(f"Error serving image: {e}")
                self.send_error(500)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/move":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length).decode("utf-8")
            params = urllib.parse.parse_qs(post_data)

            folder_name = params.get("folder", [""])[0]
            target = params.get("target", [""])[0]

            if folder_name and target:
                src = BASE_DIR / folder_name
                if target == "convertible":
                    dst = CONVERTIVE_DIR / folder_name
                elif target == "truck":
                    dst = TRUCK_DIR / folder_name
                else:
                    dst = None

                if dst and src.exists():
                    print(f"Moving {src.name} to {dst.parent.name}")
                    shutil.move(str(src), str(dst))

            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


if __name__ == "__main__":
    print(f"Starting server at http://localhost:{PORT}")
    print("Open this URL in your browser to classify images.")
    with ReusableTCPServer(("", PORT), ClassifierHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
