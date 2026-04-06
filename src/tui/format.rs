pub(crate) fn sanitize_for_display(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\u{1b}' => {
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            '\n' | '\t' => result.push(c),
            c if c.is_control() => {}
            c => result.push(c),
        }
    }
    result
}
