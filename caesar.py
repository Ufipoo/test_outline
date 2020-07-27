from tkinter import  *
class Block:
    def __init__(self, master):
        self.e0 = Entry(master, width=20) #ключ
        self.e1 = Entry(master, width=20) #кодинг
        self.e2 = Entry(master, width=20) #декодинг
        self.b1 = Button(master, text="Закодировать")
        self.b2 = Button(master, text="Декадировать")
        self.l1 = Label(master, bg='black', fg='white') #закодированное сообщение
        self.l2 = Label(master, bg='white', fg='black', width=20, text='Введите ключ')
        self.l3 = Label(master, bg='white', fg='black', width=30, text='Введите текст для кодировки')
        self.l4 = Label(master, bg='white', fg='black', width=30, text='Код')

        self.l5 = Label(master, bg='white', fg='black', width=30, text='Введите текст для расшифровки')
        self.l6 = Label(master, bg='white', fg='black', width=30, text='Расшифрованное сообщение')
        self.l7 = Label(master, bg='black', fg='white') #расшифрованное сообщение

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

    def setFunc(self, func1, func2):
        self.b1['command'] = eval('self.' + func1)
        self.b2['command'] = eval('self.' + func2)

    def coding(self):
        n = int(self.e0.get())
        alpha = [' ', 'abcdefghijklmnopqrstuvwxyz', 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя', '0123456789',
                 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
        s = self.e1.get()
        print(s)
        res = ''
        print(res)

        for c in s:
            letter = ''
            for i in range(0, 6):
                if c in alpha[i]:
                    letter = alpha[i].index(c)
                    print(letter)
                    res += alpha[i][(letter + n) % len(alpha[i])]
        self.l1['text'] = ''.join(res)

    def decoding(self):
        n = int(self.e0.get())
        alpha = [' ', 'abcdefghijklmnopqrstuvwxyz', 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя', '0123456789',
                 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
        s = self.e2.get()
        print(s)
        res = ''
        print(res)

        for c in s:
            letter = ''
            for i in range(0, 6):
                if c in alpha[i]:
                    letter = alpha[i].index(c)
                    print(letter)
                    res += alpha[i][(letter - n) % len(alpha[i])]
        self.l7['text'] = ''.join(res)


if __name__ == '__main__':
    root = Tk()
    root.title('Шифр Цезаря')
    root.resizable(False, False)
    app = Block(root)
    app.setFunc('coding', 'decoding')
    root.mainloop()
