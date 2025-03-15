import sys
import asyncio
import io
import contextlib

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QScrollArea, QCheckBox, QFileDialog, QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal

# Import your agent and LLM
from langchain_openai import ChatOpenAI
from browser_use import Agent

# Initialize the LLM (global)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
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
        # Redirect both stdout and stderr so all console output is piped to the text area.
        stream = EmittingStream(self.output)
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            asyncio.run(self.run_agent(self.prompt))
        # Emit finished signal with a final message.
        self.finished.emit("Agent finished executing.\n")
        
    async def run_agent(self, prompt):
        agent = Agent(task=prompt, llm=llm)
        await agent.run(max_steps=12)

# Main window UI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QA Methis")
        self.resize(1200, 600)
        self.history_entries = []  # Stores dicts of history entries
        self.initUI()
        
    def initUI(self):
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        
        mainLayout = QHBoxLayout()
        centralWidget.setLayout(mainLayout)
        
        # LEFT: Output area and prompt input
        leftLayout = QVBoxLayout()
        mainLayout.addLayout(leftLayout, 3)
        
        self.outputText = QTextEdit()
        self.outputText.setReadOnly(True)
        leftLayout.addWidget(self.outputText)
        
        promptLayout = QHBoxLayout()
        self.inputLine = QLineEdit()
        self.modelCombo = QComboBox()
        self.modelCombo.addItems(["gpt-4o-mini", "gpt-4o"])
        self.sendButton = QPushButton("Send")
        self.sendButton.clicked.connect(self.handleSend)
        promptLayout.addWidget(self.inputLine)
        promptLayout.addWidget(self.modelCombo)
        promptLayout.addWidget(self.sendButton)
        leftLayout.addLayout(promptLayout)
        
        # RIGHT: History panel
        rightLayout = QVBoxLayout()
        mainLayout.addLayout(rightLayout, 1)
        
        self.historyWidget = QWidget()
        self.historyLayout = QVBoxLayout()
        self.historyWidget.setLayout(self.historyLayout)
        
        self.historyScroll = QScrollArea()
        self.historyScroll.setWidgetResizable(True)
        self.historyScroll.setWidget(self.historyWidget)
        rightLayout.addWidget(self.historyScroll)
        
        # Buttons for Save, Clear (unchecked), Clear All Checkboxes and Enable All Checkboxes.
        buttonLayout = QHBoxLayout()
        self.saveButton = QPushButton("Save")
        self.clearButton = QPushButton("Clear")
        self.clearAllButton = QPushButton("Clear All")
        self.enableAllButton = QPushButton("Enable All")
        
        self.saveButton.clicked.connect(self.saveHistory)
        self.clearButton.clicked.connect(self.clearUnchecked)
        self.clearAllButton.clicked.connect(self.clearAllCheckboxes)
        self.enableAllButton.clicked.connect(self.enableAllCheckboxes)
        
        buttonLayout.addWidget(self.saveButton)
        buttonLayout.addWidget(self.clearButton)
        buttonLayout.addWidget(self.clearAllButton)
        buttonLayout.addWidget(self.enableAllButton)
        rightLayout.addLayout(buttonLayout)
        
    def handleSend(self):
        prompt = self.inputLine.text().strip()
        if not prompt:
            return
        
        # Update the global llm with the selected model from the combo box.
        selected_model = self.modelCombo.currentText()
        global llm
        llm = ChatOpenAI(
            model=selected_model,
            temperature=0.7,
        )
        
        # Create a history entry with a check box and a read-only text field (using QTextEdit for wrapping).
        historyEntry = QWidget()
        entryLayout = QHBoxLayout()
        historyEntry.setLayout(entryLayout)
        
        checkbox = QCheckBox()
        checkbox.setChecked(True)  # Checkboxes are checked by default
        promptDisplay = QTextEdit()
        promptDisplay.setPlainText(prompt)
        promptDisplay.setReadOnly(True)
        promptDisplay.setFixedHeight(50)  # Adjust as needed
        
        entryLayout.addWidget(checkbox)
        entryLayout.addWidget(promptDisplay)
        self.historyLayout.addWidget(historyEntry)
        
        # Save the history entry for later processing.
        self.history_entries.append({
            'widget': historyEntry,
            'checkbox': checkbox,
            'promptDisplay': promptDisplay
        })
        
        self.inputLine.clear()
        
        # Launch the agent code in a worker thread and connect its signals.
        self.worker = Worker(prompt)
        self.worker.output.connect(self.updateOutput)
        self.worker.finished.connect(self.displayFinished)
        self.worker.start()
        
    def updateOutput(self, text):
        # Append received text to the outputText
        self.outputText.append(text)
        
    def displayFinished(self, result):
        self.outputText.append(result)
        
    def saveHistory(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save History", "", "Text Files (*.txt);;All Files (*)"
        )
        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    for entry in self.history_entries:
                        if entry['checkbox'].isChecked():
                            # Get the text from the QTextEdit
                            f.write(entry['promptDisplay'].toPlainText() + "\n")
            except Exception as e:
                print("Error saving history:", e)
                
    def clearUnchecked(self):
        # Remove only entries with unchecked checkboxes.
        remaining_entries = []
        for entry in self.history_entries:
            if not entry['checkbox'].isChecked():
                entry['widget'].setParent(None)
            else:
                remaining_entries.append(entry)
        self.history_entries = remaining_entries
        
    def clearAllCheckboxes(self):
        # Uncheck all history entry checkboxes.
        for entry in self.history_entries:
            entry['checkbox'].setChecked(False)
            
    def enableAllCheckboxes(self):
        # Check all history entry checkboxes.
        for entry in self.history_entries:
            entry['checkbox'].setChecked(True)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
