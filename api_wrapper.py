from iqoptionapi.stable_api import IQ_Option
import pandas as pd
import time

from config import *


class ApiData:

    def __init__(self, email, password, checkConnection=True, type='PRACTICE'):
        self.email = email
        self.password = password
        self.api = IQ_Option(self.email, self.password)
        self.api.connect()
        self.api.change_balance(type)  # PRACTICE / REAL
        self.checkConnection = checkConnection

    def connectionCheck(self):
        if not self.api.check_connect() and self.checkConnection:
            print('Conexão Inválida. Tentando novamente...')
            print(self.api)
            self.api.connect()
            self.api.change_balance(type)

    def getBalance(self):
        return self.api.get_balance()

    def getResult(self):
        result_message = self.api.get_optioninfo_v2(1)['msg']['closed_options'][0]
        message = result_message['win']
        bet_size = result_message['amount']
        win_amount = result_message['win_amount']
        return message, bet_size, win_amount

    def getActiveStatus(self, active):
        open_actives = self.api.get_all_open_time()
        return open_actives['turbo'][active]['open']

    def buy(self, active, direction, amount):
        '''
        Faz um call ou put de acordo com os parâmetros passados
        '''
        complete, id = self.api.buy(amount, active, direction, 5)

        if complete:
            print('Compra realizada com sucesso!')
            print(f'ID da compra: {id}\n')
            return id
        else:
            print('Erro ao realizar compra!')
            return None

    def getHistoricalDataFrame(self, active, duration, limit):
        '''
        Retorna um dataframe com os dados históricos do ativo consultado, com a duração e limite de linhas definidos
        '''
        return pd.DataFrame(self.api.get_candles(active, duration, limit, time.time()))

    def getProfit(self,active):
        '''
        Retorna o % de lucro do ativo consultado, no modo turbo
        '''
        return self.api.get_all_profit()[active]['turbo']

# teste = ApiData(IQ_USER, IQ_PASSWORD, type='PRACTICE')
# teste.api.get_optioninfo_v2(1)
# teste.api.get_order(10830012907)