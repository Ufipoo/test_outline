import tkinter as tk

class Main(tk.Frame):
    def __init__(self, root):
        super().__init__(root)

        self.init_main()

    def init_main(self):
        toolbar = tk.Frame(bg='#483D8B', bd=2, height=100)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        shifr_caesar = tk.Button(toolbar, text='Шифр Цезаря', command=self.open_caesar, bg='#E9967A', bd=0,
                                 compound=tk.CENTER)
        shifr_caesar.pack(side=tk.TOP)

        shifr_vigenere = tk.Button(toolbar, text='Шифр Виженера', command=self.open_vigenere, bg='#ADD8E6', bd=0,
                                 compound=tk.CENTER)
        shifr_vigenere.pack(side=tk.TOP)

    def open_caesar(self):
        Caesar()
    def open_vigenere(self):
        Vigenere()


    def setFunc(self, func1, func2):
        self.b1['command'] = eval('self.' + func1)
        self.b2['command'] = eval('self.' + func2)

class Caesar(tk.Toplevel):
    def __init__(self):
        super().__init__(root)
        self.init_caesar()
        self.Block()

    def init_caesar(self):
        self.title('Шифр Цезаря')
        self.resizable(False, False)

        self.grab_set()
        self.focus_set()

    def Block(self):
        self.e0 = tk.Entry(self, width=20) #ключ
        self.e1 = tk.Entry(self, width=20) #кодинг
        self.e2 = tk.Entry(self, width=20) #декодинг
        self.b1 = tk.Button(self, text="Закодировать", command=self.caesar_coding)
        self.b2 = tk.Button(self, text="Декадировать", command=self.caesar_decoding)

        self.l1 = tk.Label(self, bg='black', fg='white') #закодированное сообщение
        self.l2 = tk.Label(self, bg='white', fg='black', width=20, text='Введите ключ')
        self.l3 = tk.Label(self, bg='white', fg='black', width=30, text='Введите текст для кодировки')
        self.l4 = tk.Label(self, bg='white', fg='black', width=30, text='Код')

        self.l5 = tk.Label(self, bg='white', fg='black', width=30, text='Введите текст для расшифровки')
        self.l6 = tk.Label(self, bg='white', fg='black', width=30, text='Расшифрованное сообщение')
        self.l7 = tk.Label(self, bg='black', fg='white') #расшифрованное сообщение

        self.l2.pack()
        self.e0.pack()

        self.l3.pack()
        self.e1.pack()
        self.b1.pack()
        self.l4.pack()
        self.l1.pack()

        self.l5.pack()
        self.e2.pack()
        self.b2.pack()
        self.l6.pack()
        self.l7.pack()

    def caesar_coding(self):
        print('cezar')
        n = int(self.e0.get())
        alpha = [' ', 'abcdefghijklmnopqrstuvwxyz', 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя', '0123456789',
                 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
        s = self.e1.get()
        #print(s)
        res = ''
        #print(res)

        for c in s:
            letter = ''
            for i in range(0, 6):
                if c in alpha[i]:
                    letter = alpha[i].index(c)
                    #print(letter)
                    res += alpha[i][(letter + n) % len(alpha[i])]
        self.l1['text'] = ''.join(res)

    def caesar_decoding(self):
        print('decoding cezar')
        n = int(self.e0.get())
        alpha = [' ', 'abcdefghijklmnopqrstuvwxyz', 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя', '0123456789',
                 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
        s = self.e2.get()
        #print(s)
        res = ''
        #print(res)

        for c in s:
            letter = ''
            for i in range(0, 6):
                if c in alpha[i]:
                    letter = alpha[i].index(c)
                    #print(letter)
                    res += alpha[i][(letter - n) % len(alpha[i])]
        self.l7['text'] = ''.join(res)

class Vigenere(tk.Toplevel):
    def __init__(self):
        super().__init__(root)
        self.init_vigenere()
        self.Block()

    def init_vigenere(self):
        self.title('Шифр Виженера')
        self.resizable(False, False)
        self.grab_set()
        self.focus_set()

    def Block(self):
        self.e0 = tk.Entry(self, width=20)  # ключ
        self.e1 = tk.Entry(self, width=20)  # кодинг
        self.e2 = tk.Entry(self, width=20)  # декодинг
        self.b1 = tk.Button(self, text="Закодировать", command=self.vigenere_coding)
        self.b2 = tk.Button(self, text="Декадировать", command=self.vigenere_decoding)

        self.l1 = tk.Label(self, bg='black', fg='white')  # закодированное сообщение
        self.l2 = tk.Label(self, bg='white', fg='black', width=20, text='Введите ключ')
        self.l3 = tk.Label(self, bg='white', fg='black', width=30, text='Введите текст для кодировки')
        self.l4 = tk.Label(self, bg='white', fg='black', width=30, text='Код')

        self.l5 = tk.Label(self, bg='white', fg='black', width=30, text='Введите текст для расшифровки')
        self.l6 = tk.Label(self, bg='white', fg='black', width=30, text='Расшифрованное сообщение')
        self.l7 = tk.Label(self, bg='black', fg='white')  # расшифрованное сообщение

        self.l2.pack()
        self.e0.pack()

        self.l3.pack()
        self.e1.pack()
        self.b1.pack()
        self.l4.pack()
        self.l1.pack()

        self.l5.pack()
        self.e2.pack()
        self.b2.pack()
        self.l6.pack()
        self.l7.pack()

    def vigenere_coding(self):
        print('vegenere')
        key = self.e0.get()
        alpha = [' ', 'abcdefghijklmnopqrstuvwxyz', 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя', '0123456789',
                 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
        s = self.e1.get()
        # print(s)
        res = ''
        # print(res)
        cod_key = []
        for c in key:  # зашифрованный ключ
            for i in range(0, 6):
                if c in alpha[i]:
                    cod_key.append(alpha[i].index(c))
        # print(cod_key)
        cod = []
        for i in range(0, len(s)):
            cod.append(cod_key[i % len(cod_key)])
        # print(cod)

        for c in range(0, len(s)):
            letter = ''
            # print(s[c])
            # print(s.index(s[c]))
            for i in range(0, 6):
                if s[c] in alpha[i]:
                    letter = alpha[i].index(s[c])  # номер буквы
                    # print(letter)
                    # print()
                    res += alpha[i][(letter + (cod[c % len(cod)])) % len(alpha[i])]
        self.l1['text'] = ''.join(res)

    def vigenere_decoding(self):
        print('decoding vegenere')
        key = self.e0.get()
        alpha = [' ', 'abcdefghijklmnopqrstuvwxyz', 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя', '0123456789',
                 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
        s = self.e2.get()
        # print(s)
        res = ''
        # print(res)
        cod_key = []
        for c in key:  # зашифрованный ключ
            for i in range(0, 6):
                if c in alpha[i]:
                    cod_key.append(alpha[i].index(c))
        # print(cod_key)
        cod = []
        for i in range(0, len(s)):
            cod.append(cod_key[i % len(cod_key)])
        # print(cod)

        for c in range(0, len(s)):
            letter = ''
            # print(s[c])
            # print(s.index(s[c]))
            for i in range(0, 6):
                if s[c] in alpha[i]:
                    letter = alpha[i].index(s[c])  # номер буквы
                    # print(letter)
                    # print()
                    res += alpha[i][(letter + len(alpha[i]) - (cod[c % len(cod)])) % len(alpha[i])]
        self.l7['text'] = ''.join(res)

if __name__ == '__main__':
    root = tk.Tk()
    root.title('1 лабараторная')
    root.resizable(False, False)
    app = Main(root)
    root.mainloop()


