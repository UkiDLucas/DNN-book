# export_music21.py
from music21 import converter, midi
from pathlib import Path

abc = Path("generated.abc").read_text(encoding="utf-8")
stream = converter.parseData(abc, format="abc")

# Optional: transpose to keep within flute range if needed
# stream = stream.transpose('P1')  # change interval as needed

mf = midi.translate.streamToMidiFile(stream)
mf.open("generated.mid", 'wb'); mf.write(); mf.close()

# To MusicXML for clean notation
stream.write('musicxml', fp='generated.musicxml')
print("Wrote generated.mid and generated.musicxml")
