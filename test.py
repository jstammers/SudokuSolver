import kivy
kivy.require('1.9.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.properties import NumericProperty, AliasProperty, ObjectProperty
from kivy.event import EventDispatcher
from kivy.uix.popup import Popup

class SudokuPosMixin(EventDispatcher):
    sudoku_x = NumericProperty(0, min=0, max=2)
    sudoku_y = NumericProperty(0, min=0, max=2)
    
    def get_sudoku_pos(self):
        return self.sudoku_x, self.sudoku_y
    def set_sudoku_pos(self, pos):
        self.sudoku_x, self.sudoku_y = pos
    sudoku_pos = AliasProperty(get_sudoku_pos, set_sudoku_pos, bind=['sudoku_x', 'sudoku_y'])

class SudokuValueMixin(EventDispatcher):
    value = NumericProperty(0, min=0, max=9)

class SudokuButton(Button, SudokuPosMixin, SudokuValueMixin):
    def __init__(self, **kwargs):
        super(SudokuButton, self).__init__(**kwargs)
        self.bind(on_release=self.show_popup)
        self.popup = SudokuSelectPopup(button=self)
    
    def show_popup(self, *args):
        self.popup.open()
    
    def set_value(self, value):
        if value == 0:
            self.value = 0
        elif self.parent.check_value(self.sudoku_pos, value):
            self.value = value
        else:
            return False
        return True

class SudokuLabel(Label, SudokuPosMixin, SudokuValueMixin):
    pass

class SudokuSelectPopup(Popup):
    button = ObjectProperty()
    
    def open(self, *args, **kwargs):
        self.error_label.text = ''
        return super(SudokuSelectPopup, self).open(*args, **kwargs)
    
    def select(self, value):
        if self.button.set_value(value):
            self.dismiss()
        else:
            self.error_label.text = 'Invalid value!'

class SudokuBox(GridLayout, SudokuPosMixin):
    def get_index(self):
        pos = self.sudoku_pos
        return pos[1] * 3 + pos[0]
    
    def load_grid(self, grid):
        for i, val in enumerate(grid):
            pos = (i // 3, i % 3)
            if val:
                self.add_widget(SudokuLabel(sudoku_pos=pos, value=val))
            else:
                self.add_widget(SudokuButton(sudoku_pos=pos))
    
    def check_value(self, pos, value):
        for cell in self.children:
            if cell.value == value:
                return False
        return self.parent.check_value(self.sudoku_pos, pos, value)
    
    def get_col(self, col):
        return [cell.value for cell in self.children if cell.sudoku_x == col]
    
    def get_row(self, row):
        return [cell.value for cell in self.children if cell.sudoku_y == row]

class SudokuGrid(GridLayout):
    def __init__(self, **kwargs):
        super(SudokuGrid, self).__init__(**kwargs)
        self.prepare_game()
    
    def check_value(self, box_pos, cell_pos, value):
        for box in self.children:
            if box.sudoku_pos == box_pos:
                continue
            elif box.sudoku_x == box_pos[0]:
                values = box.get_col(cell_pos[0])
                if value in values:
                    return False
            elif box.sudoku_y == box_pos[1]:
                values = box.get_row(cell_pos[1])
                if value in values:
                    return False
        return True
    
    def box_at(self, x, y):
        for box in self.children:
            if box.sudoku_pos == (x, y):
                return box
        return None
    
    def prepare_game(self):
        # Sudoku ID: 527
        # http://www.counton.org/sudoku/sudoku.php?id=527
        game = [[5, 7, 3, 0, 0, 0, 2, 0, 0],
                [0, 6, 0, 1, 0, 3, 0, 0, 5],
                [8, 1, 0, 0, 0, 0, 0, 0, 4],
                [4, 0, 0, 7, 9, 0, 0, 0, 0],
                [0, 0, 7, 0, 1, 4, 9, 0, 0],
                [1, 0, 0, 0, 3, 0, 7, 0, 2],
                [3, 0, 7, 9, 0, 0, 0, 8, 0],
                [5, 0, 0, 0, 7, 8, 2, 0, 0],
                [0, 8, 0, 0, 2, 0, 9, 0, 5]]
        for box in self.children:
            box.load_grid(game[box.get_index()])

root = Builder.load_string('''
<SudokuSelectPopup>:
    size_hint: 0.4, 0.5
    error_label: error_label
    
    BoxLayout:
        orientation: 'vertical'
        
        GridLayout:
            cols: 3
        
            Button:
                text: '1'
                on_press: root.select(1)
            Button:
                text: '2'
                on_press: root.select(2)
            Button:
                text: '3'
                on_press: root.select(3)
            Button:
                text: '4'
                on_press: root.select(4)
            Button:
                text: '5'
                on_press: root.select(5)
            Button:
                text: '6'
                on_press: root.select(6)
            Button:
                text: '7'
                on_press: root.select(7)
            Button:
                text: '8'
                on_press: root.select(8)
            Button:
                text: '9'
                on_press: root.select(9)
            Widget
            Button:
                text: 'Clear'
                on_press: root.select(0)
            Widget
        
        Label:
            id: error_label
            size_hint_y: None
            height: 64
<SudokuButton,SudokuLabel>:
    text: str(self.value) if self.value else ''
<SudokuBox>:
    cols: 3
    
    canvas.before:
        Color:
            rgba: 0.1, 0.1, 0.1, 1
        Rectangle:
            pos: self.pos
            size: self.size
<SudokuGrid>:
    cols: 3
    spacing: 8
    
    canvas.before:
        Color:
            rgba: 0.3, 0.3, 0.5, 1
        Rectangle:
            pos: self.pos
            size: self.size
    
    SudokuBox:
        sudoku_pos: 0, 0
    SudokuBox:
        sudoku_pos: 1, 0
    SudokuBox:
        sudoku_pos: 2, 0
    
    SudokuBox:
        sudoku_pos: 0, 1
    SudokuBox:
        sudoku_pos: 1, 1
    SudokuBox:
        sudoku_pos: 2, 1
    
    SudokuBox:
        sudoku_pos: 0, 2
    SudokuBox:
        sudoku_pos: 1, 2
    SudokuBox:
        sudoku_pos: 2, 2
SudokuGrid
''')

class TestApp(App):
    def build(self):
        return root

if __name__ == '__main__':
    TestApp().run()