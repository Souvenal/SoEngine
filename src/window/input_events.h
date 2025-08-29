#pragma once

enum class EventSource {
    Keyboard,
    Mouse
};

class InputEvent {
public:
    InputEvent(EventSource source);

    EventSource getSource() const noexcept;

private:
    EventSource source;
};

enum class KeyCode {
    Unknown,
    Space,
    Apostrophe,    // '
    Comma,         // ,
    Minus,         // -
    Period,        // .
    Slash,         // /
    Num0,
    Num1,
    Num2,
    Num3,
    Num4,
    Num5,
    Num6,
    Num7,
    Num8,
    Num9,
    Semicolon,     // ;
    Equal,         // =
    A, B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    LeftBracket,   // [
    RightBracket,  // ]
    Backslash,     
    GraveAccent,   // `
    Escape,
    Enter,
    Tab,
    Backspace,
    Insert,
    Delete,
    Right,
    Left,
    Down,
    Up,
    PageUp,
    PageDown,
    Home,
    End,
    CapsLock,
    ScrollLock,
    NumLock,
    PrintScreen,
    Pause,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
    LeftShift,
    LeftControl,
    LeftAlt,
    LeftSuper,
    RightShift,
    RightControl,
    RightAlt,
    RightSuper,
    Menu
};

enum class KeyAction {
    Down,
    Up,
    Repeat,
    Unknown
};

class KeyInputEvent : public InputEvent {
public:
    KeyInputEvent(KeyCode key, KeyAction action);

    KeyCode getCode() const noexcept;
    KeyAction getAction() const noexcept;

private:
    KeyCode key;
    KeyAction action;
};

enum class MouseButton {
    Left,
    Right,
    Middle,
    Back,
    Forward,
    Unknown
};

enum class MouseAction {
    Down,
    Up,
    Move,
    Unknown
};

class MouseButtonInputEvent : public InputEvent {
public:
    MouseButtonInputEvent(MouseButton button, MouseAction action, float posX, float posY);

    MouseButton getButton() const noexcept;
    MouseAction getAction() const noexcept;
    float getPosX() const noexcept;
    float getPosY() const noexcept;

private:
    MouseButton button;
    MouseAction action;

    float posX;
    float posY;
};