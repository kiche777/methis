# Standard library imports
import asyncio
import contextlib
import io
import json
import os
import sys
import threading

# Third-party imports
from openai import OpenAI
from PyQt5.QtCore import QSettings, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QPainter, QTextCursor
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QPushButton, 
    QScrollArea, QSplitter, QSplitterHandle, QTextBrowser, QTextEdit, 
    QVBoxLayout, QWidget
)

# Local application imports
from browser_use import Agent, AgentHistoryList, Browser, BrowserConfig
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

def get_browser(headless, profile=False, connect=False, port="9123"):
    if connect:
        cdp_url = f"http://localhost:{port}"
        config = BrowserConfig(
            headless=headless,
            disable_security=False,
            cdp_url=cdp_url
        )
    elif profile:
        chrome_instance_path = "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
        config = BrowserConfig(
            headless=headless,
            disable_security=False,
            chrome_instance_path=chrome_instance_path
        )
    else:
        config = BrowserConfig(
            headless=headless,
            disable_security=False
        )
    return Browser(config=config)

browser = get_browser(False, True, False, "9123")

# Initialize the LLM (global)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# A custom stream that sends text via a signal.
class EmittingStream(io.StringIO):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def write(self, text):
        if text:
            self.signal.emit(text)
            
    def flush(self):
        pass

# Worker thread to run the asynchronous agent code
class Worker(QThread):
    finished = pyqtSignal(str)
    output = pyqtSignal(str)  # Signal to update text output in real time
    
    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt
        
    def run(self):
        # Access maxStepsField from the Worker instance
        max_steps_field = getattr(self, 'maxStepsField', None)
        
        # Initialize output_str as an instance variable
        self.output_str = ""        
        
        # TODO: Toggle True, False to enable console output displaying in outputText or to console. Currently doesn't display nice so set to False.
        if getattr(self, 'capture_output', False):
            # Set up an OS-level pipe to intercept subprocess output.
            r_fd, w_fd = os.pipe()
            # Save the original file descriptors so that they can be restored later.
            original_stdout_fd = os.dup(1)
            original_stderr_fd = os.dup(2)
            # Redirect stdout and stderr (at OS level) to the write end of the pipe.
            os.dup2(w_fd, 1)
            os.dup2(w_fd, 2)

            # Start a background thread that reads from the pipe and emits the text.
            def pipe_reader():
                with os.fdopen(r_fd) as pipe:
                    for line in iter(pipe.readline, ""):
                        self.output.emit(line)
            reader_thread = threading.Thread(target=pipe_reader, daemon=True)
            reader_thread.start()
        else:
            # When capturing is disabled, do nothing or add alternative behavior.
            pass

        # Also redirect Python-level stdout and stderr using EmittingStream.
        stream = EmittingStream(self.output)
        stream = EmittingStream(self.output)
        # Use redirection only if capture_output is enabled.
        # TODO: Toggle True/False depending if stdout should display in outputText 
        #       - it doesn't look useful and best displayed in the console
        if getattr(self, 'capture_output', False):
            cm_out = contextlib.redirect_stdout(stream)
            cm_err = contextlib.redirect_stderr(stream)
        else:
            cm_out = contextlib.nullcontext()
            cm_err = contextlib.nullcontext()
        with cm_out, cm_err:
            try:
                # asyncio.run(self.run_agent(self.prompt))
                
                # Replace the above line with the following to run the agent in thread
                def start():
                    """Start the agent in a separate thread"""
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    history = None
                    actions = None
                    thoughts = None
                    errors = None
                    try:
                        # Display a loading overlay in the outputText box.
                        self.output.emit(
                            "<div style='position: absolute; z-index: 100; top: 0; left: 0; width: 100%; height: 100%; "
                            "background-color: rgba(0, 0, 0, 0.5); color: white; font-size: 24px; display: flex; "
                            "align-items: center; justify-content: center;'>Agent Task Running...</div>"
                        )
                        history = loop.run_until_complete(self.run_agent(self.prompt))
                                            
                        errors = history.errors()
                        if errors:
                            error_output = '<br>'.join(str(e) for e in errors)
                            self.output_str += "<br><h2>Errors</h2><br><pre>" + error_output + "</pre><br><br>"
                        
                        actions = history.model_actions()
                        if actions:
                            # Improve the output so it's easier to read for Actions.
                            self.output_str += "<br><h2>Model Actions</h2><br>"
                            for i, action in enumerate(actions, start=1):
                                self.output_str += f"<h3 style='color:blue;'>Action {i}:</h3>"
                                for key, value in action.items():
                                    if isinstance(value, dict):
                                        formatted_value = json.dumps(
                                            value,
                                            indent=2,
                                            default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o)
                                        )
                                        self.output_str += f"<b>{key}:</b> <pre>{formatted_value}</pre><br>"
                                    else:
                                        self.output_str += f"<b>{key}:</b> {value}<br>"
                                self.output_str += "</div><br>"
                        
                        thoughts = history.model_thoughts()
                        if thoughts:
                            # Improve the output so it's easier to read for Thoughts.
                            self.output_str += "<br><h2>Thoughts</h2><br>"
                            for i, thought in enumerate(thoughts, start=1):
                                self.output_str += f"<div style='margin-left: 20px;'>"
                                self.output_str += f"<span style='color:blue;'>Thought {i}:</span><br>"
                                self.output_str += f"Evaluation Previous Goal: {thought.evaluation_previous_goal}<br>"
                                self.output_str += f"<b>Memory:</b> {thought.memory}<br>"
                                self.output_str += f"<b>Next Goal:</b> {thought.next_goal}<br>"
                                self.output_str += "</div><br>"
                                self.output_str += f"<div style='margin-left: 0px;'>"
                        
                        final = history.final_result()
                        if final:
                            self.output_str += "<br><h2>Final Result</h2><br>"
                            self.output_str += "<div style='margin-left: 20px; color:purple;'>"
                            self.output_str += f"{final}<br><br>"
                            self.output_str += "</div><br>"
                        self.output.emit(self.output_str if self.output_str else "No result")
                        
                        # Convert duration (in seconds) to mm:ss format and update executionTimeLabel.
                        duration = history.total_duration_seconds()
                        minutes, seconds = divmod(int(duration), 60)
                        formatted_time = f"{minutes:02d}:{seconds:02d}"
                        # Emit the formatted execution time so the main window can update its executionTimeLabel.
                        self.output.emit(f"Execution Time: {formatted_time}")
                        
                    finally:
                        # This doesn't appear to do anything... trying to resolve where after the first process is run, running a second time throws an exception.
                        loop.run_until_complete(loop.shutdown_asyncgens())
                        loop.close()
                start()
    
            except Exception as e:
                self.output.emit("\nException Encountered:\n" + str(e))
            finally:
                # Clean up: close the write end and restore original file descriptors.
                # TODO: Uncomment to pipe subprocess console output to outputText (Main text window).
                # os.close(w_fd)
                # os.dup2(original_stdout_fd, 1)
                # os.dup2(original_stderr_fd, 2)
                
                try:
                    # Only execute AI validation if the checkbox is checked
                    if getattr(self, 'aiValidationCheckBox', None) and self.aiValidationCheckBox.isChecked():
                        # Define an async function to make the OpenAI API call
                        async def get_completion():
                            client = OpenAI()
                            completion = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "user", 
                                     "content": self.validationTextBox.toPlainText() + "\n\nThe agent prompt:\n" + self.prompt + "\n\nThe result content to evaluate:\n" + self.output_str}
                                ]
                            )
                            # Extract just the message content and display it nicely
                            message_content = completion.choices[0].message.content
                            formatted_output = f"<br><br><div style='background-color: #f0f7ff; padding: 10px; border-left: 4px solid #0066cc; margin: 10px 0;'><h3>AI Validation:</h3>{message_content}</div><br><br>"
                            self.output.emit(formatted_output)
                        
                        # Run the async function
                        asyncio.run(get_completion())
                except Exception as e:
                    self.output.emit(f"\nError: {e}")  # Emit the error message to the outputText.
                finally:                
                    self.finished.emit("Agent finished executing.\n")
                    
    async def run_agent(self, prompt):
        self.agent = Agent(task=prompt, llm=llm, browser=browser, tool_calling_method="json_mode")
        # When not capturing history, use the following line to run the agent.
        # await self.agent.run(max_steps=12)
        # Access maxStepsField from the Worker instance
        max_steps_field = getattr(self, 'maxStepsField', None)
        max_steps = int(max_steps_field.text()) if max_steps_field else 12
        history: AgentHistoryList = await self.agent.run(max_steps=max_steps)

        return history   # Return the history object

# Custom splitter handle to paint handle area green when a panel is collapsed.
class CustomSplitterHandle(QSplitterHandle):
    def paintEvent(self, event):
        # Get current sizes of the widgets in the splitter.
        sizes = self.splitter().sizes()
        # Threshold to decide if a panel is collapsed.
        threshold = 30
        if any(s <= threshold for s in sizes):
            painter = QPainter(self)
            painter.fillRect(self.rect(), Qt.green)
        else:
            super().paintEvent(event)

# Custom splitter that uses the CustomSplitterHandle.
class CustomSplitter(QSplitter):
    def createHandle(self):
        return CustomSplitterHandle(self.orientation(), self)

# Main window UI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Methis")
        self.resize(1200, 600)
        # Set the application icon
        app_icon = os.path.join(os.path.dirname(os.path.abspath(__file__)), "methis.png")
        if os.path.exists(app_icon):
            self.setWindowIcon(QIcon(app_icon))
        else:
            print(f"Icon file not found: {app_icon}")
        self.history_entries = []  # Runtime history entries (widgets)
        self.historyFile = "history.json"  # Persistent file path
        self.worker = None
        self.initUI()
        self.loadPersistedHistory()
        
    def initUI(self):
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        
        mainLayout = QVBoxLayout()
        centralWidget.setLayout(mainLayout)
        
        # Use CustomSplitter instead of QSplitter.
        splitter = CustomSplitter(Qt.Horizontal)
        mainLayout.addWidget(splitter)
        
        # LEFT: Main output and prompt input
        leftWidget = QWidget()
        # Prevent the left panel from being collapsed by setting a minimum (and optional maximum) width.
        leftWidget.setMinimumWidth(300)
        leftWidget.setMaximumWidth(1200)
        leftLayout = QVBoxLayout(leftWidget)
        self.outputLabel = QLabel("Output")
        leftLayout.addWidget(self.outputLabel)
        
        self.outputText = QTextBrowser()
        self.outputText.setAcceptRichText(True)
        self.outputText.setOpenExternalLinks(False)
        self.outputText.setReadOnly(True)
        leftLayout.addWidget(self.outputText)

        # Add Save and Clear buttons aligned to the right below outputText
        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch()  # Push buttons to the right
        self.outputSaveButton = QPushButton("Save Output")
        self.outputClearButton = QPushButton("Clear Output")
        self.outputSaveButton.clicked.connect(self.handleOutputSave)
        self.outputClearButton.clicked.connect(self.handleOutputClear)
        buttonLayout.addWidget(self.outputSaveButton)
        buttonLayout.addWidget(self.outputClearButton)
        leftLayout.addLayout(buttonLayout)
        
        # Row 1: Input Line and Control buttons
        inputButtonLayout = QHBoxLayout()
        self.inputLine = QTextEdit()
        self.inputLine.setFixedHeight(60)  # Approximately three rows tall
        self.inputLine.setLineWrapMode(QTextEdit.WidgetWidth)  # Enable word wrapping
        self.inputLine.setPlaceholderText("Enter your prompt here...")
        
        self.sendButton = QPushButton("Send")
        self.sendButton.clicked.connect(self.handleSend)
        self.pauseButton = QPushButton("Pause")
        self.pauseButton.clicked.connect(self.handlePause)
        self.resumeButton = QPushButton("Resume")
        self.resumeButton.clicked.connect(self.handleResume)
        self.stopButton = QPushButton("Stop")
        self.stopButton.clicked.connect(self.handleCancel)
        
        inputButtonLayout.addWidget(self.inputLine)
        inputButtonLayout.addWidget(self.sendButton)
        inputButtonLayout.addWidget(self.pauseButton)
        inputButtonLayout.addWidget(self.resumeButton)
        inputButtonLayout.addWidget(self.stopButton)
        
        leftLayout.addLayout(inputButtonLayout)
        
        # Row 2: Advanced Settings
        advancedLayout = QHBoxLayout()
        
        # Save selection for next app load (convinience) - Developer - Kiche777, Project/App Name - Methis
        settings = QSettings("kiche777", "Methis_App")
        saved_index = settings.value("selected_model_index", 0, int)
        self.modelCombo = QComboBox()
        self.modelCombo.addItems(["gpt-4o-mini", "gpt-4o", "ollama"])
        self.modelCombo.setCurrentIndex(saved_index)
        self.modelCombo.currentIndexChanged.connect(lambda index: settings.setValue("selected_model_index", index))
        self.modelCombo.setToolTip("Select the model to use for the agent.")
        advancedLayout.addWidget(self.modelCombo)

        self.maxStepsLabel = QLabel("Max Steps")
        self.maxStepsField = QLineEdit()
        self.maxStepsField.setText("12")
        self.maxStepsField.setMaxLength(3)
        self.maxStepsField.setFixedWidth(40)
        self.maxStepsField.setToolTip("Set the maximum number of steps for the agent to run.")
        # Create a container for maxSteps controls
        maxStepsContainer = QWidget()
        maxStepsLayout = QHBoxLayout(maxStepsContainer)
        maxStepsLayout.setContentsMargins(0, 0, 0, 0)
        maxStepsLayout.setSpacing(2)
        maxStepsLayout.addWidget(self.maxStepsLabel)
        maxStepsLayout.addWidget(self.maxStepsField)
        maxStepsLayout.addStretch()  # Add stretch to push widgets to the left  
        # Add container right after headlessCheckBox
        advancedLayout.addWidget(maxStepsContainer)

        # Create a container for AI validation checkbox with chevron and textbox
        aiValidationContainer = QWidget()
        aiValidationLayout = QVBoxLayout(aiValidationContainer)
        aiValidationLayout.setContentsMargins(0, 0, 0, 0)
        
        # Header with checkbox and chevron
        headerContainer = QWidget()
        headerLayout = QHBoxLayout(headerContainer)
        headerLayout.setContentsMargins(0, 0, 0, 0)
        headerLayout.setSpacing(2)
        
        self.aiValidationCheckBox = QCheckBox("Enable AI Validation")
        self.aiValidationCheckBox.setChecked(True)  # Default to checked
        self.aiValidationCheckBox.setToolTip("Enable AI validation to evaluate the execution output.")
        
        self.notesChevron = QPushButton("▶")  # Right-pointing triangle as chevron
        self.notesChevron.setFixedWidth(20)
        self.notesChevron.setStyleSheet("border: none; text-align: left;")
        
        headerLayout.addWidget(self.aiValidationCheckBox)
        headerLayout.addWidget(self.notesChevron)
        headerLayout.addStretch()  # Add stretch to push elements to the left
        
        # Add the header container to the validation layout
        aiValidationLayout.addWidget(headerContainer)
        
        # Notes text box (initially hidden)
        self.validationTextBox = QTextEdit()
        self.validationTextBox.setPlaceholderText("Add validation prompt here...")
        self.validationTextBox.setText('From the following content, did it meet the expectations of the prompt? Provide a response Failed, Inconlusive,Passing-With Questions,Conclusive Pass.')
        self.validationTextBox.setToolTip("Enter the prompt to evaluate the execution output.\nPrompt will also include action prompt and output content for evaluation.")
        self.validationTextBox.setFixedHeight(100)
        self.validationTextBox.hide()  # Initially hidden
        aiValidationLayout.addWidget(self.validationTextBox)
        
        # Add to the advanced layout
        advancedLayout.addWidget(aiValidationContainer)
        
        # Connect chevron click to toggle notes visibility
        def toggle_validationTextBox():
            if self.validationTextBox.isHidden():
                self.validationTextBox.show()
                self.notesChevron.setText("▼")  # Down-pointing triangle
            else:
                self.validationTextBox.hide()
                self.notesChevron.setText("▶")  # Right-pointing triangle
        
        self.notesChevron.clicked.connect(toggle_validationTextBox)

        self.headlessCheckBox = QCheckBox("Headless")
        self.headlessCheckBox.setChecked(False)
        self.headlessCheckBox.setToolTip("Run the browser in headless mode.")
        advancedLayout.addWidget(self.headlessCheckBox)
        
        self.profileCheckBox = QCheckBox("Use Browser Profile")
        self.profileCheckBox.setChecked(False)
        self.profileCheckBox.setToolTip("Use your existing browser profile with settings.\nThis helps bypass certain prompts and logins if remembered in your profile.")
        advancedLayout.addWidget(self.profileCheckBox)
        
        self.connectExistingCheckBox = QCheckBox("Connect to Browser Instance")
        self.connectExistingCheckBox.setToolTip("Connect to an existing browser instance using the remote-debugging-port.")
        self.portLabel = QLabel("Port")
        self.portField = QLineEdit()
        self.portField.setText("9123")
        self.portField.setMaxLength(5)
        self.portField.setFixedWidth(60)
        self.portField.setToolTip("Enter the remote-debugging-port.")
        advancedLayout.addWidget(self.connectExistingCheckBox)
        advancedLayout.addWidget(self.portField)

        # Define slot functions for checkbox logic
        def on_profile_toggled(checked):
            if checked:
                # When Profile is checked, uncheck Connect to Browser and Headless.
                self.connectExistingCheckBox.blockSignals(True)
                self.headlessCheckBox.blockSignals(True)
                self.connectExistingCheckBox.setChecked(False)
                self.headlessCheckBox.setChecked(False)
                self.connectExistingCheckBox.blockSignals(False)
                self.headlessCheckBox.blockSignals(False)

        def on_connect_toggled(checked):
            if checked:
            # When Connect to Browser is checked, uncheck Profile and Headless.
                self.profileCheckBox.blockSignals(True)
                self.headlessCheckBox.blockSignals(True)
                self.profileCheckBox.setChecked(False)
                self.headlessCheckBox.setChecked(False)
                self.profileCheckBox.blockSignals(False)
                self.headlessCheckBox.blockSignals(False)

        def on_headless_toggled(checked):
            if checked:
            # Headless can only be checked if Profile is enabled and Connect is disabled.
                if not self.profileCheckBox.isChecked():
                    self.profileCheckBox.blockSignals(True)
                    self.profileCheckBox.setChecked(True)
                    self.profileCheckBox.blockSignals(False)
                if self.connectExistingCheckBox.isChecked():
                    self.connectExistingCheckBox.blockSignals(True)
                    self.connectExistingCheckBox.setChecked(False)
                    self.connectExistingCheckBox.blockSignals(False)

        # Connect signals
        self.headlessCheckBox.toggled.connect(on_headless_toggled)
        self.profileCheckBox.toggled.connect(on_profile_toggled)
        self.connectExistingCheckBox.toggled.connect(on_connect_toggled)
        
        advancedContainer = QVBoxLayout()
        advancedLabel = QLabel("Advanced Settings")
        advancedContainer.addWidget(advancedLabel)
        advancedContainer.addLayout(advancedLayout)
        
        leftLayout.addLayout(advancedContainer)
        
        splitter.addWidget(leftWidget)
        
        # Set the initial sizes so that the prompt history (right panel)
        # starts with a fixed default width.
        splitter.setSizes([800, 400])
        # Prevent the right panel (index 1) from being collapsible.
        # Remove setCollapsible call to avoid out of range error
        # splitter.setCollapsible(1, False)
        
        # RIGHT: History panel
        rightWidget = QWidget()
        rightLayout = QVBoxLayout(rightWidget)
        
        # Create a horizontal layout for the "Prompt History" label and "Run All" button.
        headerLayout = QHBoxLayout()
        historyLabel = QLabel("Prompt History")
        headerLayout.addWidget(historyLabel)
        headerLayout.addStretch()  # add space between the label and the button
        runAllButton = QPushButton("Run All")
        headerLayout.addWidget(runAllButton)
        rightLayout.addLayout(headerLayout)
        
        # Create the history list widget
        self.historyList = QListWidget()
        self.historyList.setDragDropMode(QListWidget.InternalMove)
        
        # Define the history reordering handler
        def onHistoryReordered():
            # Rebuild the history_entries list to match the current visual order
            reordered_entries = []
            for i in range(self.historyList.count()):
                item = self.historyList.item(i)
                # Find the corresponding entry in the original list
                for entry in self.history_entries:
                    if entry['item'] == item:
                        reordered_entries.append(entry)
                        break
            # Update the history entries with the new order
            self.history_entries = reordered_entries
            # Update persisted history
            self.updatePersistedHistory()
            
        # Connect the model's rowsMoved signal to update history after drag and drop
        self.historyList.model().rowsMoved.connect(onHistoryReordered)
        rightLayout.addWidget(self.historyList)

        # Buttons for Save, Clear, and Toggle All Checkboxes.
        buttonLayout = QHBoxLayout()
        self.saveButton = QPushButton("Save")
        self.clearButton = QPushButton("Clear")
        self.toggleAllButton = QPushButton("Toggle All")
        
        # Connect the "Run All" button to combine all history prompts and place them on the input line.
        def run_all_history():
            combined_text = "\n".join(entry['promptDisplay'].toPlainText() for entry in self.history_entries)
            self.inputLine.setPlainText(combined_text)
        
        runAllButton.clicked.connect(run_all_history)
        
        self.saveButton.clicked.connect(self.saveHistory)
        self.clearButton.clicked.connect(self.clearChecked)
        self.toggleAllButton.clicked.connect(self.toggleAllCheckboxes)
        
        buttonLayout.addWidget(self.saveButton)
        buttonLayout.addWidget(self.clearButton)
        buttonLayout.addWidget(self.toggleAllButton)
        rightLayout.addLayout(buttonLayout)
        
        splitter.addWidget(rightWidget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
    def addHistoryEntry(self, prompt, checked=False):
        # Create a custom widget to contain the history entry
        historyEntry = QWidget()
        entryLayout = QHBoxLayout(historyEntry)
        entryLayout.setContentsMargins(2, 2, 2, 2)
        
        checkbox = QCheckBox()
        checkbox.setChecked(checked)  # Use passed argument for checkbox state
        promptDisplay = QTextEdit()
        promptDisplay.setPlainText(prompt)
        promptDisplay.setReadOnly(False)  # Now editable
        promptDisplay.setFixedHeight(50)
        promptDisplay.setAlignment(Qt.AlignTop)  # Align text to the top
        
        entryLayout.addWidget(checkbox)
        entryLayout.addWidget(promptDisplay)
        
        # Create a list widget item and set its size hint
        item = QListWidgetItem()
        item.setSizeHint(historyEntry.sizeHint())
        
        # Add the item to the list widget
        self.historyList.addItem(item)
        self.historyList.setItemWidget(item, historyEntry)

        record = {
            'item': item,
            'widget': historyEntry,
            'checkbox': checkbox,
            'promptDisplay': promptDisplay
        }
        self.history_entries.append(record)
        
        # Override focusOutEvent so that when the user clicks away after editing,
        # the app saves the current state and persists history.
        originalFocusOut = promptDisplay.focusOutEvent
        def newFocusOut(event):
            originalFocusOut(event)
            self.updatePersistedHistory()
        promptDisplay.focusOutEvent = newFocusOut

        # When the promptDisplay is double clicked, copy its text into the inputLine.
        def onDoubleClick(event):
            self.inputLine.setPlainText(promptDisplay.toPlainText())
        promptDisplay.mouseDoubleClickEvent = onDoubleClick
        
        return record

    def updatePersistedHistory(self):
        records = []
        for entry in self.history_entries:
            records.append({
                "prompt": entry['promptDisplay'].toPlainText(),
                "checked": entry['checkbox'].isChecked()
            })
        try:
            with open(self.historyFile, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
        except Exception as e:
            print("Error writing history:", e)
        
    def loadPersistedHistory(self):
        if os.path.exists(self.historyFile):
            try:
                with open(self.historyFile, "r", encoding="utf-8") as f:
                    records = json.load(f)
                for rec in records:
                    self.addHistoryEntry(rec.get("prompt", ""), rec.get("checked", True))
            except Exception as e:
                print("Error loading history:", e)
        
    def handleSend(self):
        prompt = self.inputLine.toPlainText().strip()
        if not prompt:
            return
        
        selected_model = self.modelCombo.currentText()
        global llm, browser
        browser = get_browser(self.headlessCheckBox.isChecked(), self.profileCheckBox.isChecked(), self.connectExistingCheckBox.isChecked(), self.portField.text())        
        if "ollama" not in selected_model:
            llm = ChatOpenAI(
                model=selected_model,
                temperature=0.7
            )
        else:
            llm=ChatOllama(
                # model="qwen2.5:14b",
                model="qwen2.5:32b-instruct-q4_K_M",
                temperature=0.7,
                num_predict=128000
            )    
            
        self.addHistoryEntry(prompt, checked=False)
        self.updatePersistedHistory()
        
        self.inputLine.clear()
        # Launch the agent code in a worker thread and connect its signals.
        self.worker = Worker(prompt)
        # Pass maxStepsField to the Worker instance
        self.worker.maxStepsField = self.maxStepsField
        # Pass aiValidationCheckBox to the Worker instance
        self.worker.aiValidationCheckBox = self.aiValidationCheckBox
        # Pass validationTextBox to the Worker instance
        self.worker.validationTextBox = self.validationTextBox
        # This is responsible for outputting messages in real time to outputText.
        self.worker.output.connect(self.updateOutput)
        self.worker.finished.connect(self.displayFinished)
        self.worker.start()
        
    def handleCancel(self):
        if self.worker and self.worker.isRunning() and hasattr(self.worker, 'agent'):
            self.worker.agent.stop()
            # TODO: Add colour formatting and style to these messages.
            self.updateOutput("Execution cancelled.\n")
        else:
            self.updateOutput("No execution running.\n")
            
    def handlePause(self):
        if self.worker and self.worker.isRunning() and hasattr(self.worker, 'agent'):
            self.worker.agent.pause()
            # TODO: Add colour formatting and style to these messages.
            self.updateOutput("Execution paused.\n")
        else:
            self.updateOutput("No execution running.\n")
            
    def handleResume(self):
        if self.worker and hasattr(self.worker, 'agent'):
            self.worker.agent.resume()
            # TODO: Add colour formatting and style to these messages.
            self.updateOutput("Execution resumed.\n")
        else:
            self.updateOutput("No execution to resume.\n")
                       
    def updateOutput(self, text):
        # Insert received HTML text to the outputText.
        self.outputText.insertHtml(text)
        self.outputText.moveCursor(QTextCursor.End)
        
    def displayFinished(self, result):
        self.outputText.append(result)
        self.worker = None
        
    def handleOutputSave(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Output",
            "",
            "Rich Text Format (*.rtf);;HTML Files (*.html);;All Files (*)",
            options=options
        )
        if filename:
            if not (filename.endswith(".rtf") or filename.endswith(".html")):
                filename += ".rtf"
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(self.outputText.toHtml())
            except Exception as e:
                print("Error saving output:", e)

    def handleOutputClear(self):
        self.outputText.clear()
        
    def saveHistory(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save History", "", "Text Files (*.txt);;All Files (*)"
        )
        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    for entry in self.history_entries:
                        f.write(entry['promptDisplay'].toPlainText() + "\n")
            except Exception as e:
                print("Error saving history:", e)
                
    def clearChecked(self):
        # Remove only entries with checked checkboxes
        # Iterate in reverse to avoid index shifting problems
        for i in range(len(self.history_entries) - 1, -1, -1):
            entry = self.history_entries[i]
            if entry['checkbox'].isChecked():
                # Remove from the list widget
                row = self.historyList.row(entry['item'])
                self.historyList.takeItem(row)
                # Remove from our tracking list
                self.history_entries.pop(i)
        
        # Update the persisted history after modifications
        self.updatePersistedHistory()
        
    def toggleAllCheckboxes(self):
        # If any checkbox is unchecked, check all; otherwise, uncheck all
        any_unchecked = any(not entry['checkbox'].isChecked() for entry in self.history_entries)
        
        # Set all checkboxes to the new state
        for entry in self.history_entries:
            entry['checkbox'].setChecked(any_unchecked)
        
        # Update persisted history after changing checkbox states
        self.updatePersistedHistory()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())