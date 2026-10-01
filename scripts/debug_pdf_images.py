import fitz
import sys

def debug(pdf_path):
    doc = fitz.open(pdf_path)
    print(f"Total pages: {len(doc)}")
    for i in range(min(10, len(doc))):
        page = doc[i]
        imgs = page.get_images(full=True)
        info = page.get_image_info(hashes=True)
        print(f"Page {i+1}: get_images={len(imgs)}, get_image_info={len(info)}")
        for idx, img in enumerate(info):
            print(f"  Img {idx}: bbox={img.get('bbox')}, size={img.get('width')}x{img.get('height')}")
    doc.close()

if __name__ == "__main__":
    debug(sys.argv[1])
