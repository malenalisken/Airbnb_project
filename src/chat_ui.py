
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from html import escape
from agent import new_chat

language = widgets.ToggleButtons(
    options=[("Norsk", "no"), ("English", "en")],
    value="no",
    description="Språk:"
)

role = widgets.ToggleButtons(
    options=[("Gjest / Guest", "Gjest"), ("Utleier / Host", "Utleier")],
    value="Gjest",
    description="Rolle:"
)

header = widgets.HTML()
history = widgets.Output(
    layout=widgets.Layout(
        border="1px solid #ddd",
        height="350px",
        overflow="auto",
        padding="12px"
    )
)

message_box = widgets.Textarea(
    layout=widgets.Layout(width="100%", height="75px")
)

send_button = widgets.Button(
    description="Send melding",
    button_style="primary",
    icon="paper-plane"
)

messages = []
chat = None


def update_labels():
    norwegian = language.value == "no"

    language.description = "Språk:" if norwegian else "Language:"
    role.description = "Rolle:" if norwegian else "Role:"
    send_button.description = "Send melding" if norwegian else "Send message"
    message_box.placeholder = (
        "Skriv spørsmålet ditt her..."
        if norwegian else "Type your question here..."
    )

    subtitle = (
        "Spør om Airbnb-priser eller sammenlign bydeler i Oslo."
        if norwegian else
        "Ask about Airbnb prices or compare Oslo neighbourhoods."
    )

    header.value = f"""
    <div style="background:#243b53;color:white;padding:18px;border-radius:12px">
        <h2 style="color:white;margin:0">🏠 Airbnb Price Assistant – Oslo</h2>
        <p style="color:#dbeafe;margin:8px 0 0">{subtitle}</p>
    </div>
    """


def show_messages():
    with history:
        clear_output(wait=True)

        for sender, content in messages:
            is_user = sender == "user"
            alignment = "right" if is_user else "left"
            color = "#dbeafe" if is_user else "#f1f5f9"
            name = (
                ("Du" if language.value == "no" else "You")
                if is_user else "AI Assistant"
            )

            display(HTML(f"""
            <div style="text-align:{alignment};margin:12px 0">
                <div style="
                    display:inline-block;
                    max-width:85%;
                    text-align:left;
                    background:{color};
                    color:#172033;
                    padding:12px 16px;
                    border-radius:14px;
                    white-space:pre-wrap;
                    overflow-wrap:anywhere;
                ">
                    <b>{name}</b>
                    <div>{escape(str(content))}</div>
                </div>
            </div>
            """))


def reset_chat(change=None):
    global chat

    messages.clear()
    chat = new_chat(role.value)
    update_labels()
    show_messages()


def send_message(button):
    if send_button.disabled:
        return

    question = message_box.value.strip()
    if not question:
        return

    message_box.value = ""
    messages.append(("user", question))
    show_messages()

    send_button.disabled = True

    try:
        selected_language = "Norwegian" if language.value == "no" else "English"

        response = chat.send_message(
            f"Please answer in {selected_language}.\n\n{question}"
        )

        answer = response.text or "Ingen svar mottatt / No response received."
        messages.append(("assistant", answer))

    except Exception as error:
        messages.append(("assistant", f"Feil / Error: {error}"))

    finally:
        send_button.disabled = False
        show_messages()


def launch_chat():
    language.observe(reset_chat, names="value")
    role.observe(reset_chat, names="value")
    send_button.on_click(send_message)

    reset_chat()

    display(widgets.VBox(
        [
            header,
            widgets.HBox([language, role]),
            history,
            message_box,
            send_button
        ],
        layout=widgets.Layout(width="90%")
    ))
