[theme]
base = "light"
primaryColor = "#10a37f"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f7f7f8"
textColor = "#111111"


/* 1. Password input box — always visible, not just on hover */
section[data-testid="stSidebar"] [data-testid="stExpander"] div[data-baseweb="input"],
section[data-testid="stSidebar"] [data-testid="stExpander"] div[data-baseweb="input"] input {
    opacity: 1 !important;
    visibility: visible !important;
}

/* 2. Sidebar collapse/close arrow button — always visible */
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
    opacity: 1 !important;
    visibility: visible !important;
    display: flex !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button {
    opacity: 1 !important;
    background-color: transparent !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button svg {
    opacity: 1 !important;
    fill: #ececec !important;
}

/* 3. Scrollbar — always visible inside sidebar */
section[data-testid="stSidebar"] ::-webkit-scrollbar {
    width: 4px !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"] ::-webkit-scrollbar-track {
    background: #171717 !important;
}

section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
    background-color: #3a3a3a !important;
    border-radius: 4px !important;
    opacity: 1 !important;
}

/* 4. All sidebar elements — force full visibility at all times */
section[data-testid="stSidebar"] * {
    opacity: 1 !important;
}