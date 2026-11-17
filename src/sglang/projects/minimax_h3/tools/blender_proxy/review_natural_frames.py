"""Contact sheets from actual full-resolution frames, for independent review."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
IDS=('01-neon-car-ad','02-cliff-hero','04-mechanical-ad','05-crystal-stage','06-motorcycle-tempo')

def main(case):
    root=ROOT/'docs/demo-brainstorm/camera-director'/case
    directory=root/'assets/scene/natural-20261108-v1'
    routes=json.loads((directory/'camera-paths.json').read_text())['routes']
    receipt=[]
    for route in routes:
        step=6 if (case,route) in [('04-mechanical-ad','outer_half_orbit'),('05-crystal-stage','stage_arc')] else 12
        samples=sorted({1,20,24,25,28,32,40,144,*range(12,145,step)})
        columns=6;rows=(len(samples)+columns-1)//columns
        sheet=Image.new('RGB',(2016,rows*212),'#111820');d=ImageDraw.Draw(sheet)
        for i,frame in enumerate(samples):
            p=root/'assets/proxy/natural-20261108-v1/fine'/f'.{route}_frames'/f'frame_{frame:04}.png'
            x=i%columns*336;y=i//columns*212
            sheet.paste(Image.open(p).resize((336,192)),(x,y+20));d.text((x,y),f'{route} | {frame:03}',fill='white')
            receipt.append(dict(route=route,style='fine',frame=frame,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        sheet.save(directory/f'dense-{route}.jpg',quality=94)
    sheet=Image.new('RGB',(2016,5*212),'#111820');d=ImageDraw.Draw(sheet)
    for j,style in enumerate(('wire','depth','coarse','fine','subject_bg')):
        for k,route in enumerate(routes):
            for i,frame in enumerate((1,72,144)):
                p=root/'assets/proxy/natural-20261108-v1'/style/f'.{route}_frames'/f'frame_{frame:04}.png'
                x=(k*3+i)*336;y=j*212
                sheet.paste(Image.open(p).resize((336,192)),(x,y+20));d.text((x,y),f'{style} {route} | {frame:03}',fill='white')
                receipt.append(dict(route=route,style=style,frame=frame,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    sheet.save(directory/'all-styles-filmstrip.jpg',quality=94)
    (directory/'review-samples.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print('REVIEW_SHEETS',case,len(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=IDS);a=p.parse_args()
    for case in ([a.case] if a.case else IDS):main(case)
