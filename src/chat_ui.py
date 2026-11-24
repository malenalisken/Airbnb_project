
"""User interface for the Oslo Airbnb Price Assistant."""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from html import escape

import markdown

from agent import new_chat, send_message_with_retry


# --------------------------------------------------
# LANGUAGE AND ROLE SETTINGS
# --------------------------------------------------

language = widgets.ToggleButtons(
    options=[("Norsk", "no"), ("English", "en")],
    value="no",
    description="Språk:"
)

role = widgets.ToggleButtons(
    options=[
        ("Gjest / Guest", "Gjest"),
        ("Utleier / Host", "Utleier")
    ],
    value="Gjest",
    description="Rolle:"
)


# --------------------------------------------------
# CHAT COMPONENTS
# --------------------------------------------------

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
    placeholder="Skriv spørsmålet ditt her...",
    layout=widgets.Layout(
        width="100%",
        height="75px"
    )
)

send_button = widgets.Button(
    description="Send melding",
    button_style="primary",
    icon="paper-plane"
)

status = widgets.HTML()

messages = []
chat = None
launched = False


# --------------------------------------------------
# UPDATE TEXT AND LANGUAGE
# --------------------------------------------------

def update_labels():

    norwegian = language.value == "no"

    language.description = (
        "Språk:" if norwegian else "Language:"
    )

    role.description = (
        "Rolle:" if norwegian else "Role:"
    )

    send_button.description = (
        "Send melding" if norwegian else "Send message"
    )

    message_box.placeholder = (
        "Skriv spørsmålet ditt her..."
        if norwegian else
        "Type your question here..."
    )

    subtitle = (
        "Spør om Airbnb-priser eller sammenlign bydeler i Oslo."
        if norwegian else
        "Ask about Airbnb prices or compare Oslo neighbourhoods."
    )

    header.value = f"""
    <div style="
        background:#243b53;
        color:white;
        padding:18px;
        border-radius:12px;
    ">
        <h2 style="color:white;margin:0">
            🏠 Airbnb Price Assistant – Oslo
        </h2>
        <p style="color:#dbeafe;margin:8px 0 0">
            {subtitle}
        </p>
    </div>
    """


# --------------------------------------------------
# FORMAT AI RESPONSES USING MARKDOWN
# --------------------------------------------------

def format_message(content):

    safe_content = escape(str(content))

    return markdown.markdown(
        safe_content,
        extensions=[
            "fenced_code",
            "tables",
            "nl2br"
        ]
    )


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

def show_messages():

    with history:

        clear_output(wait=True)

        for sender, content in messages:

            is_user = sender == "user"

            alignment = "right" if is_user else "left"

            color = (
                "#dbeafe" if is_user else "#f1f5f9"
            )

            name = (
                ("Du" if language.value == "no" else "You")
                if is_user else "AI Assistant"
            )

            formatted_content = format_message(content)

            display(HTML(f"""
            <div style="
                text-align:{alignment};
                margin:12px 0;
            ">
                <div style="
                    display:inline-block;
                    max-width:85%;
                    text-align:left;
                    background:{color};
                    color:#172033;
                    padding:12px 16px;
                    border-radius:14px;
                    overflow-wrap:anywhere;
                ">
                    <b>{name}</b>
                    <div style="
                        margin-top:8px;
                        line-height:1.5;
                    ">
                        {formatted_content}
                    </div>
                </div>
            </div>
            """))


# --------------------------------------------------
# START OR RESET CONVERSATION
# --------------------------------------------------

def reset_chat(change=None):

    global chat

    messages.clear()

    chat = new_chat(role.value)

    status.value = ""

    update_labels()
    show_messages()


# --------------------------------------------------
# LANGUAGE CHANGE
# --------------------------------------------------

def change_language(change):

    update_labels()

    # Keep existing conversation and history.
    show_messages()


# --------------------------------------------------
# SEND MESSAGE
# --------------------------------------------------

def send_message(button):

    if send_button.disabled:
        return

    question = message_box.value.strip()

    if not question:
        return

    if chat is None:
        return

    # Add user message
    message_box.value = ""

    messages.append(("user", question))

    show_messages()

    # Disable input while Gemini is working
    send_button.disabled = True
    message_box.disabled = True

    norwegian = language.value == "no"

    status.value = (
        "<p style='color:#64748b'>"
        "AI-assistenten skriver ..."
        "</p>"
        if norwegian else
        "<p style='color:#64748b'>"
        "AI assistant is typing ..."
        "</p>"
    )

    try:

        selected_language = (
            "Norwegian" if norwegian else "English"
        )

        prompt = (
            f"Please answer in {selected_language}.\n\n"
            f"{question}"
        )

        # Use retry handling from agent.py
        response = send_message_with_retry(
            chat,
            prompt
        )

        answer = response.text

        if not answer:
            answer = (
                "Jeg mottok ikke noe svar. "
                "Prøv igjen."
                if norwegian else
                "No response received. Please try again."
            )

        messages.append(("assistant", answer))

    except Exception as error:

        error_code = getattr(error, "code", None)
        error_text = str(error)

        if error_code == 429 or "429" in error_text:

            answer = (
                "Gemini har nådd API-kvoten. "
                "Vent til Google tillater nye forespørsler, "
                "og prøv igjen."
                if norwegian else
                "Gemini has reached its API quota. "
                "Please wait until Google allows new requests."
            )

        elif error_code == 503 or "503" in error_text:

            answer = (
                "Gemini er midlertidig utilgjengelig. "
                "Prøv igjen om litt."
                if norwegian else
                "Gemini is temporarily unavailable. "
                "Please try again shortly."
            )

        else:

            answer = (
                "Noe gikk galt. Prøv igjen om litt."
                if norwegian else
                "Something went wrong. Please try again shortly."
            )

        messages.append(("assistant", answer))

    finally:

        send_button.disabled = False
        message_box.disabled = False

        status.value = ""

        show_messages()


# --------------------------------------------------
# LAUNCH CHAT INTERFACE
# --------------------------------------------------

def launch_chat():

    global launched

    if launched:
        print("Chatten er allerede startet.")
        return

    language.observe(
        change_language,
        names="value"
    )

    role.observe(
        reset_chat,
        names="value"
    )

    send_button.on_click(send_message)

    reset_chat()

    display(
        widgets.VBox(
            [
                header,
                widgets.HBox([language, role]),
                history,
                status,
                message_box,
                send_button
            ],
            layout=widgets.Layout(width="90%")
        )
    )

    launched = True




