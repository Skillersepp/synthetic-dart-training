"""
Background Augmentation

Ersetzt den transparenten Hintergrund der Dartboard-Bilder durch
zufällige Bilder aus einem Hintergrund-Datensatz.

Die Dartboard-Bilder müssen PNG mit Alpha-Kanal sein.
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from tqdm import tqdm
import shutil


class BackgroundAugmentor:
    """
    Fügt zufällige Hintergründe zu transparenten Dartboard-Bildern hinzu.
    """

    def __init__(
        self,
        background_dir: Path,
        output_size: Tuple[int, int] = (800, 800),
        offset_enabled: bool = False,
        max_offset_fraction: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Args:
            background_dir: Ordner mit Hintergrundbildern
            output_size: Ausgabe-Größe (width, height)
            offset_enabled: Dartboard zufällig verschieben (deaktiviert)
            max_offset_fraction: Maximale Verschiebung als Bruchteil der Bildgröße
            seed: Random Seed für Reproduzierbarkeit
        """
        self.background_dir = Path(background_dir)
        self.output_size = output_size
        self.offset_enabled = offset_enabled
        self.max_offset_fraction = max_offset_fraction

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Hintergrundbilder laden
        self.background_paths = self._find_backgrounds()
        print(f"Gefunden: {len(self.background_paths)} Hintergrundbilder")

        if len(self.background_paths) == 0:
            raise ValueError(
                f"Keine Hintergrundbilder gefunden in: {self.background_dir}\n"
                f"Unterstützte Formate: .jpg, .jpeg, .png, .webp"
            )

    def _find_backgrounds(self) -> List[Path]:
        """Findet alle Hintergrundbilder im Ordner (rekursiv)."""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG', '*.PNG']
        paths = []
        for ext in extensions:
            paths.extend(self.background_dir.rglob(ext))
        return sorted(paths)

    def load_random_background(self) -> np.ndarray:
        """Lädt ein zufälliges Hintergrundbild und skaliert es."""
        bg_path = random.choice(self.background_paths)
        bg = cv2.imread(str(bg_path))

        if bg is None:
            # Fallback: Einfarbiger Hintergrund
            print(f"Warnung: Konnte {bg_path} nicht laden, nutze grauen Hintergrund")
            bg = np.full((*self.output_size[::-1], 3), 128, dtype=np.uint8)
            return bg

        # Auf Zielgröße skalieren (mit crop wenn nötig)
        bg = self._resize_and_crop(bg, self.output_size)

        return bg

    def _resize_and_crop(
        self,
        img: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Skaliert und croppt ein Bild auf die Zielgröße.
        Behält das Seitenverhältnis bei und croppt mittig.
        """
        target_w, target_h = target_size
        h, w = img.shape[:2]

        # Skalierungsfaktor berechnen (so dass Bild mindestens Zielgröße hat)
        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Skalieren
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Mittig croppen
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        img = img[start_y:start_y + target_h, start_x:start_x + target_w]

        return img

    def composite(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        offset: Tuple[int, int] = (0, 0)
    ) -> np.ndarray:
        """
        Kombiniert Vordergrund (mit Alpha) und Hintergrund.

        Args:
            foreground: BGRA Bild (4 Kanäle)
            background: BGR Bild (3 Kanäle)
            offset: (x, y) Verschiebung des Vordergrunds

        Returns:
            Kombiniertes BGR Bild
        """
        fg_h, fg_w = foreground.shape[:2]
        bg_h, bg_w = background.shape[:2]

        # Offset anwenden
        ox, oy = offset

        # Ergebnis-Bild (Kopie des Hintergrunds)
        result = background.copy()

        # Bereich berechnen wo Vordergrund platziert wird
        # Vordergrund-Bereich
        fg_x1 = max(0, -ox)
        fg_y1 = max(0, -oy)
        fg_x2 = min(fg_w, bg_w - ox)
        fg_y2 = min(fg_h, bg_h - oy)

        # Hintergrund-Bereich
        bg_x1 = max(0, ox)
        bg_y1 = max(0, oy)
        bg_x2 = bg_x1 + (fg_x2 - fg_x1)
        bg_y2 = bg_y1 + (fg_y2 - fg_y1)

        if fg_x2 <= fg_x1 or fg_y2 <= fg_y1:
            # Kein Überlappungsbereich
            return result

        # Alpha-Kanal extrahieren und normalisieren
        alpha = foreground[fg_y1:fg_y2, fg_x1:fg_x2, 3:4] / 255.0

        # Vordergrund RGB
        fg_rgb = foreground[fg_y1:fg_y2, fg_x1:fg_x2, :3]

        # Hintergrund-Bereich
        bg_region = result[bg_y1:bg_y2, bg_x1:bg_x2]

        # Alpha-Blending
        blended = (fg_rgb * alpha + bg_region * (1 - alpha)).astype(np.uint8)
        result[bg_y1:bg_y2, bg_x1:bg_x2] = blended

        return result

    def augment_image(
        self,
        image_path: Path,
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Augmentiert ein einzelnes Bild mit zufälligem Hintergrund.

        Args:
            image_path: Pfad zum PNG mit Transparenz
            output_path: Optional Ausgabepfad

        Returns:
            Augmentiertes Bild (BGR)
        """
        # Vordergrund laden (mit Alpha)
        fg = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if fg is None:
            raise ValueError(f"Konnte Bild nicht laden: {image_path}")

        # Prüfen ob Alpha-Kanal vorhanden
        if fg.shape[2] != 4:
            print(f"Warnung: {image_path} hat keinen Alpha-Kanal, überspringe")
            return cv2.imread(str(image_path))

        # Auf Zielgröße skalieren falls nötig
        if fg.shape[:2] != self.output_size[::-1]:
            fg = cv2.resize(fg, self.output_size, interpolation=cv2.INTER_LINEAR)

        # Zufälligen Hintergrund laden
        bg = self.load_random_background()

        # Offset berechnen (falls aktiviert)
        offset = (0, 0)
        if self.offset_enabled:
            max_offset_x = int(self.output_size[0] * self.max_offset_fraction)
            max_offset_y = int(self.output_size[1] * self.max_offset_fraction)
            offset = (
                random.randint(-max_offset_x, max_offset_x),
                random.randint(-max_offset_y, max_offset_y)
            )

        # Compositing
        result = self.composite(fg, bg, offset)

        # Speichern falls gewünscht
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), result)

        return result

    def augment_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        num_variations: int = 1,
        copy_labels: bool = True,
        labels_dir: Optional[Path] = None
    ) -> dict:
        """
        Augmentiert einen ganzen Datensatz.

        Args:
            input_dir: Ordner mit Original-Bildern (PNG mit Transparenz)
            output_dir: Ausgabe-Ordner
            num_variations: Anzahl Variationen pro Bild
            copy_labels: Labels mitkopieren
            labels_dir: Ordner mit Labels (default: input_dir/../labels)

        Returns:
            Statistiken
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Labels-Ordner
        if labels_dir is None:
            labels_dir = input_dir.parent / 'labels'

        # Output-Ordner erstellen
        output_images = output_dir / 'images'
        output_labels = output_dir / 'labels'
        output_images.mkdir(parents=True, exist_ok=True)
        if copy_labels:
            output_labels.mkdir(parents=True, exist_ok=True)

        # Alle PNG-Dateien finden
        image_paths = sorted(list(input_dir.glob('*.png')) + list(input_dir.glob('*.PNG')))
        print(f"Gefunden: {len(image_paths)} Bilder")

        stats = {
            'input_images': len(image_paths),
            'output_images': 0,
            'skipped': 0
        }

        for img_path in tqdm(image_paths, desc="Augmenting"):
            stem = img_path.stem

            for var_idx in range(num_variations):
                # Output-Name
                if num_variations > 1:
                    out_name = f"{stem}_var{var_idx:03d}.png"
                else:
                    out_name = f"{stem}.png"

                out_path = output_images / out_name

                try:
                    # Augmentieren
                    self.augment_image(img_path, out_path)
                    stats['output_images'] += 1

                    # Label kopieren
                    if copy_labels:
                        # JSON Label
                        json_label = labels_dir / f"{stem}.json"
                        if json_label.exists():
                            out_label_name = out_name.replace('.png', '.json').replace('.PNG', '.json')
                            shutil.copy2(json_label, output_labels / out_label_name)

                        # TXT Label (YOLO Format)
                        txt_label = labels_dir / f"{stem}.txt"
                        if txt_label.exists():
                            out_label_name = out_name.replace('.png', '.txt').replace('.PNG', '.txt')
                            shutil.copy2(txt_label, output_labels / out_label_name)

                except Exception as e:
                    print(f"Fehler bei {img_path}: {e}")
                    stats['skipped'] += 1

        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Background Augmentation für Dartboard-Bilder'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Ordner mit Original-Bildern (PNG mit Transparenz)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Ausgabe-Ordner'
    )
    parser.add_argument(
        '--backgrounds', '-b',
        type=str,
        required=True,
        help='Ordner mit Hintergrundbildern'
    )
    parser.add_argument(
        '--variations', '-v',
        type=int,
        default=1,
        help='Anzahl Variationen pro Bild (default: 1)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=800,
        help='Ausgabe-Größe in Pixeln (default: 800)'
    )
    parser.add_argument(
        '--offset',
        action='store_true',
        help='Dartboard zufällig verschieben (deaktiviert by default)'
    )
    parser.add_argument(
        '--max-offset',
        type=float,
        default=0.1,
        help='Maximale Verschiebung als Bruchteil (default: 0.1 = 10%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random Seed (default: 42)'
    )
    parser.add_argument(
        '--labels',
        type=str,
        default=None,
        help='Labels-Ordner (default: input/../labels)'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Labels nicht kopieren'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Background Augmentation")
    print("=" * 60)
    print(f"Input:        {args.input}")
    print(f"Output:       {args.output}")
    print(f"Backgrounds:  {args.backgrounds}")
    print(f"Variations:   {args.variations}")
    print(f"Size:         {args.size}x{args.size}")
    print(f"Offset:       {'Enabled' if args.offset else 'Disabled'}")
    print("=" * 60)

    # Augmentor erstellen
    augmentor = BackgroundAugmentor(
        background_dir=Path(args.backgrounds),
        output_size=(args.size, args.size),
        offset_enabled=args.offset,
        max_offset_fraction=args.max_offset,
        seed=args.seed
    )

    # Datensatz augmentieren
    stats = augmentor.augment_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        num_variations=args.variations,
        copy_labels=not args.no_labels,
        labels_dir=Path(args.labels) if args.labels else None
    )

    print("\n" + "=" * 60)
    print("Fertig!")
    print("=" * 60)
    print(f"Input Bilder:  {stats['input_images']}")
    print(f"Output Bilder: {stats['output_images']}")
    print(f"Übersprungen:  {stats['skipped']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
