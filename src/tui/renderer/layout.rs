#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Rect {
    pub x: u16,
    pub y: u16,
    pub width: u16,
    pub height: u16,
}

impl Rect {
    pub const fn new(x: u16, y: u16, width: u16, height: u16) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RootLayoutMode {
    Wide,
    Compact,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LayoutPlan {
    pub mode: RootLayoutMode,
    pub top_bar: Rect,
    pub transcript: Rect,
    pub composer: Rect,
}

const TOP_BAR_HEIGHT: u16 = 1;

pub(crate) fn layout_for(width: u16, height: u16, composer_height: u16) -> LayoutPlan {
    let mode = if width >= 100 && height >= 28 {
        RootLayoutMode::Wide
    } else {
        RootLayoutMode::Compact
    };

    let top_bar = Rect::new(0, 0, width, TOP_BAR_HEIGHT);
    let content_y = TOP_BAR_HEIGHT;
    let mut content_height = height.saturating_sub(TOP_BAR_HEIGHT);

    let composer = Rect::new(
        0,
        height.saturating_sub(composer_height),
        width,
        composer_height,
    );
    content_height = content_height.saturating_sub(composer_height);

    let transcript_y = content_y;
    let transcript_height = content_height;
    LayoutPlan {
        mode,
        top_bar,
        transcript: Rect::new(0, transcript_y, width, transcript_height),
        composer,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_switches_to_compact_for_small_terminal() {
        let plan = layout_for(80, 24, 5);
        assert_eq!(plan.mode, RootLayoutMode::Compact);
    }

    #[test]
    fn layout_keeps_single_column_for_large_terminal() {
        let plan = layout_for(140, 40, 5);
        assert_eq!(plan.mode, RootLayoutMode::Wide);
        assert_eq!(plan.transcript.width, 140);
        assert_eq!(plan.transcript.y, 1);
        assert_eq!(plan.top_bar.height, 1);
        assert_eq!(plan.composer.y, 35);
    }
}
