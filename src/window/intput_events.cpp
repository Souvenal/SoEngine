#include "input_events.h"

// InputEvent
InputEvent::InputEvent(EventSource source)
    : source(source) {}

EventSource InputEvent::getSource() const noexcept {
    return source;
}

// KeyInputEvent
KeyInputEvent::KeyInputEvent(KeyCode key, KeyAction action)
    : InputEvent(EventSource::Keyboard), key(key), action(action) {}

KeyCode KeyInputEvent::getCode() const noexcept {
    return key;
}

KeyAction KeyInputEvent::getAction() const noexcept {
    return action;
}

// MouseButtonInputEvent
MouseButtonInputEvent::MouseButtonInputEvent(MouseButton button, MouseAction action, float posX, float posY)
    : InputEvent(EventSource::Mouse), button(button), action(action), posX(posX), posY(posY)
{}

MouseButton MouseButtonInputEvent::getButton() const noexcept {
    return button;
}

MouseAction MouseButtonInputEvent::getAction() const noexcept {
    return action;
}

float MouseButtonInputEvent::getPosX() const noexcept {
    return posX;
}

float MouseButtonInputEvent::getPosY() const noexcept {
    return posY;
}