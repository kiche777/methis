import sys
import asyncio
import io
import contextlib

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QScrollArea, QCheckBox, QFileDialog, QComboBox, QSplitter, QSplitterHandle
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPainter

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
            try:
                # asyncio.run(self.run_agent(self.prompt))
                
                # Replace the above line with the following to run the agent in thread
                def start():
                    """Start the agent in a separate thread"""
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.run_agent(self.prompt))
                start()
    
            except Exception as e:
                self.output.emit("Execution cancelled.\n")
        # Emit finished signal with a final message.
        self.finished.emit("Agent finished executing.\n")
        
    async def run_agent(self, prompt):
        self.agent = Agent(task=prompt, llm=llm)
        await self.agent.run(max_steps=12)

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
        self.setWindowTitle("QA Methis")
        self.resize(1200, 600)
        self.history_entries = []  # Stores dicts of history entries
        self.worker = None
        self.initUI()
        
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
        
        self.pauseButton = QPushButton("Pause")
        self.pauseButton.clicked.connect(self.handlePause)
        promptLayout.addWidget(self.pauseButton)
        
        self.resumeButton = QPushButton("Resume")
        self.resumeButton.clicked.connect(self.handleResume)
        promptLayout.addWidget(self.resumeButton)
        
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.clicked.connect(self.handleCancel)
        promptLayout.addWidget(self.cancelButton)
        
        leftLayout.addLayout(promptLayout)
        splitter.addWidget(leftWidget)
        
        # Set the initial sizes so that the prompt history (right panel)
        # starts with a fixed default width.
        splitter.setSizes([800, 400])
        # Prevent the right panel (index 1) from being collapsible.
        splitter.setCollapsible(1, False)
        
        # RIGHT: History panel
        rightWidget = QWidget()
        rightLayout = QVBoxLayout(rightWidget)
        
        self.historyWidget = QWidget()
        self.historyLayout = QVBoxLayout(self.historyWidget)
        self.historyWidget.setLayout(self.historyLayout)
        
        self.historyScroll = QScrollArea()
        self.historyScroll.setWidgetResizable(True)
        self.historyScroll.setWidget(self.historyWidget)
        rightLayout.addWidget(self.historyScroll)
        
        # Buttons for Save, Clear, Clear All and Enable All Checkboxes.
        buttonLayout = QHBoxLayout()
        self.saveButton = QPushButton("Save")
        self.clearButton = QPushButton("Clear")
        self.clearAllButton = QPushButton("Clear All")
        self.enableAllButton = QPushButton("Enable All")
        
        # Redirect standard output and error to the outputText widget.
        class ConsoleOutput(io.StringIO):
            def __init__(self, callback):
                super().__init__()
                self.callback = callback
            def write(self, text):
                if text:
                    self.callback(text)
            def flush(self):
                pass
        sys.stdout = ConsoleOutput(self.updateOutput)
        sys.stderr = ConsoleOutput(self.updateOutput)
        
        self.saveButton.clicked.connect(self.saveHistory)
        self.clearButton.clicked.connect(self.clearUnchecked)
        self.clearAllButton.clicked.connect(self.clearAllCheckboxes)
        self.enableAllButton.clicked.connect(self.enableAllCheckboxes)
        
        buttonLayout.addWidget(self.saveButton)
        buttonLayout.addWidget(self.clearButton)
        buttonLayout.addWidget(self.clearAllButton)
        buttonLayout.addWidget(self.enableAllButton)
        rightLayout.addLayout(buttonLayout)
        
        splitter.addWidget(rightWidget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
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
        
        # Create a history entry with a check box and a read-only text field.
        historyEntry = QWidget()
        entryLayout = QHBoxLayout()
        historyEntry.setLayout(entryLayout)
        checkbox = QCheckBox()
        checkbox.setChecked(True)  # Checkboxes are checked by default
        promptDisplay = QTextEdit()
        promptDisplay.setPlainText(prompt)
        promptDisplay.setReadOnly(True)
        promptDisplay.setFixedHeight(50)  # Adjust as needed

        # When the promptDisplay is double clicked, copy its text into the inputLine.
        def onDoubleClick(event):
            self.inputLine.setText(promptDisplay.toPlainText())
        promptDisplay.mouseDoubleClickEvent = onDoubleClick

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
        
    def handleCancel(self):
        if self.worker and self.worker.isRunning() and hasattr(self.worker, 'agent'):
            # self.worker.terminate()
            # self.worker.wait()
            self.worker.agent.stop()
            self.updateOutput("Execution cancelled.\n")
        else:
            self.updateOutput("No execution running.\n")
            
            
    def handlePause(self):
        if self.worker and self.worker.isRunning() and hasattr(self.worker, 'agent'):
            # self.worker.terminate()
            self.worker.agent.pause()
            self.updateOutput("Execution paused.\n")
        else:
            self.updateOutput("No execution running.\n")
            
    def handleResume(self):
        if self.worker and hasattr(self.worker, 'agent'):
            # self.worker.start()
            self.worker.agent.resume()
            self.updateOutput("Execution resumed.\n")
        else:
            self.updateOutput("No execution to resume.\n")
                       
    
    def updateOutput(self, text):
        # Append received text to the outputText.
        self.outputText.append(text)
        
    def displayFinished(self, result):
        self.outputText.append(result)
        self.worker = None
        
    def saveHistory(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save History", "", "Text Files (*.txt);;All Files (*)"
        )
        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    for entry in self.history_entries:
                        if entry['checkbox'].isChecked():
                            # Get the text from the QTextEdit.
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
