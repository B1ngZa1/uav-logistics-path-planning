"""Render downloaded PDFs and generate contact sheets for visual review."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent.parent
preview = Path(__file__).resolve().parent / '预览'
preview.mkdir(exist_ok=True)
pdfs = sorted(root.rglob('*.pdf'))

def render(item):
    i, path = item
    prefix = preview / f'PDF{i:02d}'
    subprocess.run(['pdftoppm','-f','1','-l','1','-scale-to','750','-singlefile','-png',str(path),str(prefix)],check=True,capture_output=True)
    return (i,path,prefix.with_suffix('.png'))

with ThreadPoolExecutor(max_workers=4) as executor:
    rendered = list(executor.map(render,enumerate(pdfs,1)))
for group in range((len(rendered)+5)//6):
    subset=rendered[group*6:(group+1)*6]
    sheet=Image.new('RGB',(1500,1620),'#dddddd')
    draw=ImageDraw.Draw(sheet)
    for pos,(i,path,png) in enumerate(subset):
        img=Image.open(png).convert('RGB')
        img.thumbnail((485,745))
        x=(pos%3)*500+(500-img.width)//2
        y=(pos//3)*810+35
        sheet.paste(img,(x,y))
        draw.text(((pos%3)*500+10,(pos//3)*810+10),f'PDF{i:02d}',fill='black')
        print(f'PDF{i:02d}: {path.relative_to(root)}')
    sheet.save(preview/f'contact_{group+1}.png')
    print('CONTACT',preview/f'contact_{group+1}.png')
low=next(root.glob('01_*/*.pdf'))
for page in (5,7,162):
    subprocess.run(['pdftoppm','-f',str(page),'-l',str(page),'-scale-to','1100','-singlefile','-png',str(low),str(preview/f'low_altitude_p{page}')],check=True,capture_output=True)
