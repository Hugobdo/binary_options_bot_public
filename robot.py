
import time
from datetime import datetime, timedelta
from tqdm import tqdm

from config import IQ_USER, IQ_PASSWORD
from api_wrapper import ApiData
from predict import ModelPredictor
from mongo_connector import MongoData

class TradingBot:

    def __init__(self, user, password, active, model_name=None, type='PRACTICE', bet_size=2):
        self.user = user
        self.password = password
        self.type = type
        self.active = active
        self.api = ApiData(self.user, self.password, type=self.type)
        self.predictor = ModelPredictor(model_name, self.active)
        self.mongo = MongoData()
        self.bet_size = bet_size
        self.actions_count = 0
        self.adjusted_this_minute = False
        self.use_confidence_level = False
        self.total_count = 0

    def is_open(self):
        """
        Check if the asset market is open. 
        If the market is closed and the asset is not an OTC asset, switch to its OTC counterpart.
        If the market is closed and the asset is an OTC asset, switch to its non-OTC counterpart.
        """
        if not self.api.getActiveStatus(self.active):

            # If the asset is not an OTC asset, append "-OTC"
            if "-OTC" not in self.active:
                self.active += "-OTC"

            # If the asset is an OTC asset, remove "-OTC"
            else:
                self.active = self.active.replace("-OTC", "")

            # After switching, check the status of the new asset. 
            # If it's also closed, you might want to handle this situation differently.
            if not self.api.getActiveStatus(self.active):
                raise Exception("Both {self.active} and its OTC counterpart are closed.")

        return True

    def predict(self):
        """
        Use the model to predict the future value of the asset.
        """

        self.predictor.get_new_data()

        if self.predictor.mode == 'lstm':
            self.predictor.predict_lstm()

        elif self.predictor.mode == 'arima':
            self.predictor.predict_arima(use_confidence_level=self.use_confidence_level)

        else:
            raise Exception("Invalid prediction mode.")

        return self.predictor.lower_bound[0], self.predictor.upper_bound[0]

    def check_price(self):
        """
        Check the current price of the asset.
        """
        current_data = self.api.api.get_candles(self.active, 60, 1, time.time())
        return current_data[0]['close']

    def check_result(self):
        """
        Retrieve the results of a given trade.
        """
        message, bet_size, win_amount = self.api.getResult()
        message = 'equal' if bet_size == win_amount else message
        return message, bet_size, win_amount
    
    def run(self):
        """
        Keeps the bot running, making predictions at the start of every minute 
        and then deciding whether to trade based on the price checked until 30 seconds into the minute.
        """
        first_run = True
        predicted_lower_bound = None
        predicted_upper_bound = None
        logs_list = []

        # Configuring tqdm to show the current second and switch moment
        pbar = tqdm(total=60, desc="Initialization", position=0, leave=True, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]')
        
        while True:
            current_second = datetime.now().second

            # Check for the first run scenario
            if first_run and current_second < 30:
                print("First run. Updating data...")
                self.predictor.get_new_data()
                time_to_wait = 60 - datetime.now().second
                switch_second = 60
                pbar.set_description("Waiting for next minute...")
                for _ in range(time_to_wait):
                    current_second = datetime.now().second
                    pbar.set_postfix_str(f"Current: {current_second}s, Switch in: {switch_second}s")
                    pbar.n = current_second
                    pbar.refresh()
                    time.sleep(1)
                pbar.reset()
                first_run = False
                continue
                
            # If the current time is the start of a new minute, make a new prediction.
            if current_second == 0:
                print("\nNew minute. Predicting...")
                predicted_lower_bound, predicted_upper_bound = self.predict()
                logs_list = []
                self.adjusted_this_minute = False

            # If the current second is less than 30, check the price and decide whether to trade.
            if current_second < 29:
                switch_second = 30
                current_price = self.check_price()

                if current_price < predicted_lower_bound and self.actions_count < 1:
                    print(f"Price {current_price} < {predicted_lower_bound}. Buying 'Call'...")
                    buy_id = self.api.buy(self.active, 'Call', self.bet_size)
                    action = 'Call'
                    self.actions_count += 1

                elif current_price > predicted_upper_bound and self.actions_count < 1:
                    print(f"Price {current_price} > {predicted_upper_bound}. Buying 'Put'...")
                    buy_id = self.api.buy(self.active, 'Put', self.bet_size)
                    action = 'Put'
                    self.actions_count += 1

                else:
                    pbar.set_description(f"CP: {current_price:.5f} | LB: {predicted_lower_bound:.5f} | UB: {predicted_upper_bound:.5f}")
                    buy_id = None
                    action = 'Pass'

                tick_log = {
                    'buy_id': buy_id,
                    'current_time': datetime.now(),
                    'end_time': datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=1),
                    'active': self.active,
                    'predictor_mode': self.predictor.mode,
                    'seq_length': self.predictor.seq_length,
                    'confidence_level': self.predictor.confidence_level,
                    'bet_size': self.bet_size,
                    'action': action,
                    'current_price': current_price,
                    'predicted_price': self.predictor.predictions_original_scale[0],
                    'predicted_upper_bound': predicted_upper_bound,
                    'predicted_lower_bound': predicted_lower_bound,
                }
                logs_list.append(tick_log)

            # Then wait for the next second.
            else:
                switch_second = 60

                # If there are any logs to save, save them.
                if len(logs_list) > 0:
                    print("Saving logs...")
                    self.mongo.log_collection.insert_many(logs_list)
                    logs_list = []
                pbar.set_description("Preparing for next minute")

                # Adjust the confidence level based on the number of actions taken.
                if not self.adjusted_this_minute and self.use_confidence_level:
                    if self.actions_count < 2 and self.predictor.confidence_level > 0.01:
                        # Aumentar o nível de confiança se ações < 2
                        self.predictor.confidence_level -= 0.01  # Diminuir em 2%
                        print(f"Decreasing confidence level to {self.predictor.confidence_level}")
                    elif self.actions_count > 2 and self.predictor.confidence_level < 1:
                        # Diminuir o nível de confiança se ações > 2
                        self.predictor.confidence_level += 0.01  # Incrementar em 2%
                        print(f"Increasing confidence level to {self.predictor.confidence_level}")

                    # Mantenha o nível de confiança entre 0 e 1
                    self.predictor.confidence_level = max(0.01, min(1, self.predictor.confidence_level))

                    # Reset actions_count para o próximo minuto
                    self.actions_count = 0
                    self.adjusted_this_minute = True

            pbar.set_postfix_str(f"Current: {current_second}s, Switch: {switch_second}s")
            pbar.n = current_second
            pbar.refresh()
            time.sleep(1)
            first_run = False

            # If we reach the end of a minute, reset the progress bar
            if current_second == 59:
                pbar.reset()
                self.total_count += self.actions_count
                self.actions_count = 0
            
            if self.total_count > 10:
                break

bot = TradingBot(IQ_USER, IQ_PASSWORD, active='EURUSD', model_name="model_20231003_205958", bet_size=100)
bot.run()