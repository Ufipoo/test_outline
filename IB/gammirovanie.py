import tkinter as tk
import numpy as np

class Main(tk.Frame):
    def __init__(self, root):
        super().__init__(root)

        self.init_main()

    def init_main(self):
        toolbar = tk.Frame(bg='#483D8B', bd=2, height=100)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        shifr_gamma = tk.Button(toolbar, text='Шифр гаммирования на основе ключа', command=self.open_gamma, bg='#E9967A', bd=0,
                                 compound=tk.CENTER)
        shifr_gamma.pack(side=tk.TOP)

        shifr_gamma2 = tk.Button(toolbar, text='Шифр гаммирование на основе псвевдослучайных бит', command=self.open_gamma2, bg='#ADD8E6', bd=0,
                                 compound=tk.CENTER)
        shifr_gamma2.pack(side=tk.TOP)


    def open_gamma(self):
        Gamma()

    def open_gamma2(self):
        Gamma2()

class Gamma(tk.Toplevel):
    def __init__(self):
        super().__init__(root)
        self.init_gamma()
        self.Block()

    def init_gamma(self):
        self.title('Шифр Гаммирования')
        self.resizable(False, False)
        self.geometry('350x350')
        self['bg'] = '#FA8072'

        self.grab_set()
        self.focus_set()

    def Block(self):
        self.e0 = tk.Entry(self, width=20) #ключ
        self.e1 = tk.Entry(self, width=20) #кодинг
        self.e2 = tk.Entry(self, width=20) #декодинг
        self.b1 = tk.Button(self, text="Закодировать", command=self.gamma_coding)
        self.b2 = tk.Button(self, text="Декодировать", command=self.gamma_decoding)

        self.l1 = tk.Label(self, bg='black', fg='white') #закодированное сообщение
        self.l2 = tk.Label(self, bg='red', fg='black', width=20, text='Введите ключ')
        self.l3 = tk.Label(self, bg='white', fg='black', width=30, text='Введите текст для кодировки')
        self.l4 = tk.Label(self, bg='white', fg='black', width=30, text='Код')

        self.l5 = tk.Label(self, bg='white', fg='black', width=30, text='Введите текст для расшифровки \n(Если поле оставить свободным, \nпо умолчанию расшифруется \nврехнее закодированное сообщение)')


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

    def gamma_coding(self):
        print('cezar')
        n = self.e0.get()
        s = self.e1.get()
        print(s)
        res = ''
        #print(res)
        word_code = ' '.join(format(ord(x), 'b') for x in s)
        key_code = ' '.join(format(ord(x), 'b') for x in n)
        print(word_code)
        print(key_code)
        a, b = int(word_code.split(' ')[0], 2), int(key_code.split(' ')[0], 2)
        print(a, b)
        print(int(bin(a ^ b), 2))

        for i in range(0, len(word_code.split(' '))):
            a, b = int(word_code.split(' ')[i], 2), int(key_code.split(' ')[i % len(key_code.split(' '))], 2)
            res += chr(int(bin(a ^ b), 2))
            print((int(bin(a ^ b), 2)))
        print(res)
        self.l1['text'] = ''.join(res)


    def gamma_decoding(self):
        print('decoding cezar')
        n = self.e0.get()
        if self.e2.get() == '':
            s = self.l1['text']
        else:
            s = self.e2.get()
        print(s)
        res = ''
        word_code = ' '.join(format(ord(x), 'b') for x in s)
        key_code = ' '.join(format(ord(x), 'b') for x in n)
        print(word_code)
        print(key_code)
        a, b = int(word_code.split(' ')[0], 2), int(key_code.split(' ')[0], 2)
        print(a, b)
        print(int(bin(a ^ b), 2))

        for i in range(0, len(word_code.split(' '))):
            a, b = int(word_code.split(' ')[i], 2), int(key_code.split(' ')[i % len(key_code.split(' '))], 2)
            res += chr(int(bin(a ^ b), 2))
            print((int(bin(a ^ b), 2)))
        print(res)
        #print(res)

        self.l7['text'] = ''.join(res)

'________________________________________________________________________'
class Gamma2(tk.Toplevel):
    def __init__(self):
        super().__init__(root)
        self.init_gamma2()
        self.Block()

    def init_gamma2(self):
        self.title('Шифр Гаммирования с псевдослучайным ключом')
        self.resizable(False, False)
        self.geometry('350x350')
        self['bg'] = '#FA8072'

        self.grab_set()
        self.focus_set()

    def Block(self):
        self.e1 = tk.Entry(self, width=20) #кодинг
        self.e2 = tk.Entry(self, width=20) #декодинг
        self.b1 = tk.Button(self, text="Закодировать", command=self.gamma_coding2)
        self.b2 = tk.Button(self, text="Декодировать", command=self.gamma_decoding2)
        self.b3 = tk.Button(self, text="Сгенерировать ключ", command=self.key_generation)

        self.l1 = tk.Label(self, bg='black', fg='white') #закодированное сообщение
        self.l2 = tk.Label(self, bg='red', fg='black', width=20, text='Сгенерированный ключ')
        self.l3 = tk.Label(self, bg='white', fg='black', width=30, text='Введите текст для кодировки')
        self.l4 = tk.Label(self, bg='white', fg='black', width=30, text='Код')
        self.l5 = tk.Label(self, bg='white', fg='black', width=30, text='Введите текст для расшифровки \n(Если поле оставить свободным, \nпо умолчанию расшифруется \nврехнее закодированное сообщение)')


        self.l6 = tk.Label(self, bg='white', fg='black', width=30, text='Расшифрованное сообщение')
        self.l7 = tk.Label(self, bg='black', fg='white') #расшифрованное сообщение
        self.l8 = tk.Label(self, bg='black', fg='white')

        self.b3.pack()
        self.l2.pack()
        self.l8.pack()

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

    def key_generation(self):
        n = np.ones(128)
        for i in range(0, 64):
            n[i] = 0
        print(n)
        np.random.shuffle(n)
        print(n)
        key_code = []
        for i in range(0, 16):
            l = ''
            for j in range(i*8, 8*(i+1)):
                l += str(np.int(n[j]))
            key_code.append(l)
        key = ''
        for i in range(0, len(key_code)):
            key += chr(int(key_code[i], 2))
        print(key)
        self.l8['text'] = ''.join(key)

    def gamma_coding2(self):
        print('cezar')
        n = self.l8['text']
        s = self.e1.get()
        print(s)
        res = ''
        #print(res)
        word_code = ' '.join(format(ord(x), 'b') for x in s)
        key_code = ' '.join(format(ord(x), 'b') for x in n)
        print(word_code)
        print(key_code)
        for i in range(0, len(word_code.split(' '))):
            print(int(word_code.split(' ')[i], 2), int(key_code.split(' ')[i % len(key_code)], 2))
            a, b = int(word_code.split(' ')[i], 2), int(key_code.split(' ')[i % len(key_code)], 2)
            res += chr(int(bin(a ^ b), 2))
            print((int(bin(a ^ b), 2)))
        print(res)
        self.l1['text'] = ''.join(res)


    def gamma_decoding2(self):
        print('decoding cezar')
        n = self.l8['text']
        if self.e2.get() == '':
            s = self.l1['text']
        else:
            s = self.e2.get()
        print(s)
        res = ''
        word_code = ' '.join(format(ord(x), 'b') for x in s)
        key_code = ' '.join(format(ord(x), 'b') for x in n)
        print(word_code)
        print(key_code)
        a, b = int(word_code.split(' ')[0], 2), int(key_code.split(' ')[0], 2)
        print(a, b)
        print(int(bin(a ^ b), 2))

        for i in range(0, len(word_code.split(' '))):
            a, b = int(word_code.split(' ')[i], 2), int(key_code.split(' ')[i % len(key_code.split(' '))], 2)
            res += chr(int(bin(a ^ b), 2))
            print((int(bin(a ^ b), 2)))
        print(res)
        #print(res)

        self.l7['text'] = ''.join(res)


if __name__ == '__main__':
    root = tk.Tk()
    root.title('2 лабараторная')
    root.resizable(False, False)
    app = Main(root)
    root.mainloop()




