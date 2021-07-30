#Token types               
#EOF = end of file
INTEGER, PLUS, EOF = 'INTEGER', 'PLUS', 'EOF'
MINUS = 'MINUS'
MULTIPLY = 'MULTIPLICATION'
DIVIDE = 'DIVISION'



class Token(object):

    def __init__(self, type, value):

        #Types can be integer, plus or eof as mentioned above. Values are digits.

        self.type = type
        self.value = value


    def __str__(self):

        #String representation of type and value of the class object
        return 'Token({type}, {value})'.format(type = self.type , value = repr(self.value))

    def __repr__(self):
        return self.__str__()



class Interpreter(object):

    def __init__(self, text):
        
        self.text = text
        self.pos = 0
        self.current_token = None
        self.current_char = self.text[self.pos]


    def error(self):

        raise Exception('Error parsing input')


    #Advancing into current char
    def advance(self):
        self.pos += 1
        if self.pos > len(self.text) -1:
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]

    def skip_whitespace(self):

        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    #Accepting multidigits as a string then returning as integer with whole.
    def integer(self):
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return int(result)


    #This method tokenize or divide the input into tokens (lexical analyzer)
    def get_next_token(self):
        
        while self.current_char is not None:

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isdigit():
                return Token(INTEGER, self.integer())

            if self.current_char == '+':
                self.advance()
                return Token(PLUS, '+')
            
            if self.current_char == '-':
                self.advance()
                return Token(MINUS, '-')

            if self.current_char == '*':
                self.advance()
                return Token(MULTIPLY, '*')
            
            if self.current_char == '/':
                self.advance()
                return Token(DIVIDE, '/')

            self.error()

        return Token(EOF, None)



    #Iterating the tokens if operators match, Parser/Interpreter code
    def eat(self, token_type):

        if self.current_token.type == token_type:
            self.current_token = self.get_next_token()

        else:
            self.error()


    def term(self):
        token = self.current_token
        self.eat(INTEGER)
        return token.value
        
    #Factor
    def factor(self):
        self.eat(INTEGER)

    #Parser and interpreter's lair.
    def expr(self):


        self.factor()
        
        while self.current_token.type in (MULTIPLY, DIVIDE):
            token = self.current_token
            if token.type == MULTIPLY:
                self.eat(MULTIPLY)
                self.factor()
            elif token.type == DIVIDE:
                self.eat(DIVIDE)
                self.factor()

        
        self.current_token = self.get_next_token()

        result = self.term()

        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            if token.type == PLUS:
                self.eat(PLUS)
                result = result + self.term()

            elif token.type == MINUS:
                self.eat(MINUS)
                result = result - self.term()



        """
        #Middle token expected to be a plus operator
        op = self.current_token
        if op.type == PLUS:
            self.eat(PLUS)
        elif op.type == MINUS:
            self.eat(MINUS)
        elif op.type == MULTIPLY:
            self.eat(MULTIPLY)
        elif op.type == DIVIDE:
            self.eat(DIVIDE)

        #Right token, expected to be an integer
        right = self.current_token
        self.eat(INTEGER)

        #Doing the actual addition operation
        if op.type == PLUS:
            result = left.value + right.value

        elif op.type == MINUS:
            result = left.value - right.value

        elif op.type == MULTIPLY:
            result = left.value * right.value

        elif op.type == DIVIDE:
            result = left.value / right.value """


        return result


def main():
    
    while True:

        try:
            text = input('calculator > ')

        except EOFError:
            break
            
        if not text:
            continue

        calc_interpreter = Interpreter(text)
        result = calc_interpreter.expr()
        print(result)




if __name__ == '__main__':
    main()

