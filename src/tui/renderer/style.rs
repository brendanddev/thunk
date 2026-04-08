use crossterm::style::Color;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub fn to_crossterm(self) -> Color {
        Color::Rgb {
            r: self.r,
            g: self.g,
            b: self.b,
        }
    }
}

const FG_SHIFT: u64 = 0;
const BG_SHIFT: u64 = 24;
const FLAG_SHIFT: u64 = 48;
const BOLD_FLAG: u64 = 1 << 0;
const DIM_FLAG: u64 = 1 << 1;
const ITALIC_FLAG: u64 = 1 << 2;
const UNDERLINE_FLAG: u64 = 1 << 3;
const REVERSE_FLAG: u64 = 1 << 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PackedStyle(pub u64);

impl PackedStyle {
    pub const fn new(fg: Rgb, bg: Rgb) -> Self {
        Self(rgb_bits(fg, FG_SHIFT) | rgb_bits(bg, BG_SHIFT))
    }

    pub const fn with_bold(mut self) -> Self {
        self.0 |= BOLD_FLAG << FLAG_SHIFT;
        self
    }

    pub const fn with_dim(mut self) -> Self {
        self.0 |= DIM_FLAG << FLAG_SHIFT;
        self
    }

    pub const fn with_italic(mut self) -> Self {
        self.0 |= ITALIC_FLAG << FLAG_SHIFT;
        self
    }

    pub const fn with_underline(mut self) -> Self {
        self.0 |= UNDERLINE_FLAG << FLAG_SHIFT;
        self
    }

    pub const fn with_reverse(mut self) -> Self {
        self.0 |= REVERSE_FLAG << FLAG_SHIFT;
        self
    }

    pub const fn fg(self) -> Rgb {
        unpack_rgb(self.0, FG_SHIFT)
    }

    pub const fn bg(self) -> Rgb {
        unpack_rgb(self.0, BG_SHIFT)
    }

    pub const fn is_bold(self) -> bool {
        self.flags() & BOLD_FLAG != 0
    }

    pub const fn is_dim(self) -> bool {
        self.flags() & DIM_FLAG != 0
    }

    pub const fn is_italic(self) -> bool {
        self.flags() & ITALIC_FLAG != 0
    }

    pub const fn is_underline(self) -> bool {
        self.flags() & UNDERLINE_FLAG != 0
    }

    pub const fn is_reverse(self) -> bool {
        self.flags() & REVERSE_FLAG != 0
    }

    const fn flags(self) -> u64 {
        self.0 >> FLAG_SHIFT
    }
}

const fn rgb_bits(rgb: Rgb, shift: u64) -> u64 {
    ((rgb.r as u64) | ((rgb.g as u64) << 8) | ((rgb.b as u64) << 16)) << shift
}

const fn unpack_rgb(bits: u64, shift: u64) -> Rgb {
    let value = (bits >> shift) & 0x00ff_ffff;
    Rgb {
        r: (value & 0xff) as u8,
        g: ((value >> 8) & 0xff) as u8,
        b: ((value >> 16) & 0xff) as u8,
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Theme {
    pub background: Rgb,
    pub border: Rgb,
    pub border_active: Rgb,
    pub text: Rgb,
    pub text_muted: Rgb,
    pub text_dim: Rgb,
    pub accent: Rgb,
    pub assistant: Rgb,
    pub warning: Rgb,
    pub danger: Rgb,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            background: Rgb::new(13, 16, 20),
            border: Rgb::new(56, 63, 72),
            border_active: Rgb::new(102, 214, 255),
            text: Rgb::new(234, 239, 244),
            text_muted: Rgb::new(170, 180, 191),
            text_dim: Rgb::new(107, 117, 127),
            accent: Rgb::new(102, 214, 255),
            assistant: Rgb::new(223, 104, 184),
            warning: Rgb::new(242, 179, 86),
            danger: Rgb::new(237, 104, 109),
        }
    }
}

impl Theme {
    pub fn base(self) -> PackedStyle {
        PackedStyle::new(self.text, self.background)
    }

    pub fn muted(self) -> PackedStyle {
        PackedStyle::new(self.text_muted, self.background)
    }

    pub fn dim(self) -> PackedStyle {
        PackedStyle::new(self.text_dim, self.background)
    }

    pub fn badge_user(self) -> PackedStyle {
        PackedStyle::new(self.accent, self.background).with_bold()
    }

    pub fn badge_assistant(self) -> PackedStyle {
        PackedStyle::new(self.assistant, self.background).with_bold()
    }

    pub fn chip_accent(self) -> PackedStyle {
        PackedStyle::new(self.accent, self.background).with_bold()
    }

    pub fn chip_warning(self) -> PackedStyle {
        PackedStyle::new(self.warning, self.background).with_bold()
    }

    pub fn chip_danger(self) -> PackedStyle {
        PackedStyle::new(self.danger, self.background).with_bold()
    }

    pub fn border(self) -> PackedStyle {
        PackedStyle::new(self.border, self.background)
    }

    pub fn border_active(self) -> PackedStyle {
        PackedStyle::new(self.border_active, self.background).with_bold()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_style_round_trips_rgb_and_flags() {
        let style = PackedStyle::new(Rgb::new(1, 2, 3), Rgb::new(4, 5, 6))
            .with_bold()
            .with_dim()
            .with_underline();
        assert_eq!(style.fg(), Rgb::new(1, 2, 3));
        assert_eq!(style.bg(), Rgb::new(4, 5, 6));
        assert!(style.is_bold());
        assert!(style.is_dim());
        assert!(style.is_underline());
        assert!(!style.is_reverse());
    }
}
