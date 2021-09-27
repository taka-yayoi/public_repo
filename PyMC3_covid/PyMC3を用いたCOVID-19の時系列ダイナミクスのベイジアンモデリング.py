# Databricks notebook source
# MAGIC %md
# MAGIC # PyMC3を用いたCOVID-19の時系列ダイナミクスのベイジアンモデリング
# MAGIC 
# MAGIC 本ノートブックでは、COVID-19の疾病パラメータを予測するためにどのようにPyMC3を用いるのかを説明します。
# MAGIC 
# MAGIC ここでは以下の処理を行います。
# MAGIC - SIRモデルのλとμを推定するためにベイジアン推論を用いる
# MAGIC - 任意の`t`におけるI(t)を推定するために上のパラメータを使用する
# MAGIC - `R0`(基本再生産数)を計算する
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/07/12</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>8.3ML</td></tr>
# MAGIC </table>
# MAGIC <img style="margin-top:25px;" src="https://sajpstorage.blob.core.windows.net/workshop20210205/databricks-logo-small-new.png" width="140">
# MAGIC 
# MAGIC **参考情報**
# MAGIC - [PyMC3を用いたCOVID\-19の時系列ダイナミクスのベイジアンモデリング \- Qiita](https://qiita.com/taka_yayoi/items/4a0316853687957f6526)
# MAGIC - [Using Bayesian Statistics and PyMC3 to Model the Temporal Dynamics of COVID\-19 \- The Databricks Blog](https://databricks.com/blog/2021/01/06/bayesian-modeling-of-the-temporal-dynamics-of-covid-19-using-pymc3.html)
# MAGIC - [SIRモデル \- Wikipedia](https://ja.wikipedia.org/wiki/SIR%E3%83%A2%E3%83%87%E3%83%AB)
# MAGIC - [aseyboldt/sunode: Solve ODEs fast, with support for PyMC3](https://github.com/aseyboldt/sunode)
# MAGIC - [Rでデータの密度を見る（カーネル密度関数・ラグプロット） \- Qiita](https://qiita-user-contents.imgix.net/https%3A%2F%2Fi.imgur.com%2FTo3Ytuz.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=bba76d47ca708cedee47d53ae939f5ad)
# MAGIC - [新型コロナウイルス 日本国内の感染者数・死者数・重症者数データ｜NHK特設サイト](https://www3.nhk.or.jp/news/special/coronavirus/data-all/)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ライブラリのインストール

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC conda install -y -c conda-forge sunode plotly pymc3

# COMMAND ----------

# MAGIC %md
# MAGIC `sympy==1.8`ではエラーとなるため1.4にします。

# COMMAND ----------

# MAGIC %pip install sympy==1.4

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.figure_factory as ff
import scipy.stats
import pymc3 as pm
import arviz as az
import sunode
import sunode.wrappers.as_theano
from pymc3.ode import DifferentialEquation
import theano.tensor as tt
import theano
import datetime
import shelve
from datetime import datetime as dt
import time

# COMMAND ----------

# MAGIC %md
# MAGIC ## COVID_dataオブジェクトの生成

# COMMAND ----------

# -------- 使用方法 --------#
# covid_obj = COVID_data('US', Population=328.2e6)
# covid_obj.get_dates(data_begin='7/11/20', data_end='7/20/20')
# sir_model = SIR_model(covid_obj)
# likelihood = {'distribution': 'lognormal', 'sigma': 2}
# prior= {'lam': 0.4, 'mu': 1/8, lambda_std', 0.5 'mu_std': 0.5 }
# sir_model.run_SIR_model(n_samples=20, n_tune=10, likelihood=likelihood)
np.random.seed(0)

class COVID_data():

    def __init__(self, country='US', Population = 328.2e6):

        # 感染者数データ
        confirmed_cases_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
        self.confirmed_cases = pd.read_csv(confirmed_cases_url, sep=',')
        
        # 死者数データ
        deaths_url =  'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
        self.deaths = pd.read_csv(deaths_url, sep=',')
        path_to_save = ''

        # ------------------------- 推定対象の国 -------------------
        self.country = country
        self.N = Population   # 国の人口
                         # ドイツ - 83.7e6
                         # アメリカ - 328.2e6
                         # 日本 - 126.3e6

    def get_dates(self, data_begin='7/11/20', data_end='7/20/20'):

        # ------------------------- 推定期間 ----------------------#
        self.data_begin = data_begin  # 昨日までのデータを取得
        self.data_end = data_end
        self.num_days_to_predict = 14
        confirmed_cases = self.confirmed_cases
        country = self.country
        self.cases_country = confirmed_cases.loc[confirmed_cases["Country/Region"] == country]
        self.cases_obs = np.array(confirmed_cases.loc[confirmed_cases["Country/Region"] == country, data_begin:data_end])[0]

        print("------------ Cases for selected period ----------- ",self.cases_obs)

        date_data_end = confirmed_cases.loc[confirmed_cases["Country/Region"] == self.country, data_begin:data_end].columns[-1]
        month, day, year = map(int,date_data_end.split('/'))
        date_data_end = datetime.date(year+2000, month, day)
        date_today = date_data_end + datetime.timedelta(days=1)
        print("------------- Cases yesterday ({}): {} and day before yesterday: {} ------------".format(date_data_end.isoformat(), *self.cases_obs[:-3:-1]))
        self.num_days = len(self.cases_obs)

        day_before_start = dt.strptime(data_end, '%m/%d/%y') + datetime.timedelta(days=-1)
        day_before_start_cases = np.array(self.cases_country.loc[:, day_before_start.strftime('%-m/%-d/%-y')])
        print("------------ Day before start and cases for that date ------------", day_before_start, day_before_start_cases)
        future_days_begin = dt.strptime(data_end, '%m/%d/%y') + datetime.timedelta(days=1)
        future_days_end = future_days_begin + datetime.timedelta(days=self.num_days_to_predict)
        self.future_days_begin_s = future_days_begin.strftime('%-m/%-d/%-y')
        self.future_days_end_s = future_days_end.strftime('%-m/%-d/%-y')
        print("------------- Future date begin and end -------------",self.future_days_begin_s, self.future_days_end_s)
        self.future_days = np.array(self.cases_country.loc[:, self.future_days_begin_s : self.future_days_end_s])[0]
        print("------------- Future days cases ------------", self.future_days)

# COMMAND ----------

# MAGIC %md
# MAGIC ## SIRモデルの生成
# MAGIC 
# MAGIC **注意** 59行目の`cores=2`の値をお使いのクラスターのコア数に合わせてください

# COMMAND ----------

class SIR_model():

    def __init__(self, covid_data) :

        # ------------------------- Covid_data オブジェクト -----------------------#
        self.covid_data = covid_data
        # ------------------------- SIRモデルのセットアップ、しかし、実行する際には明示的に呼び出す必要があります ------------------------#
        self.setup_SIR_model()

    def SIR_non_normalized(self, y, t, p):
        ds = -p[0] * y[0] * y[1] /self.covid_data.N
        di = p[0] * y[0] * y[1] / self.covid_data.N  -  p[1] * y[1]
        return [ds, di]

    def setup_SIR_model(self):
        self.time_range = np.arange(0,len(self.covid_data.cases_obs),1)
        self.I0 = self.covid_data.cases_obs[0]
        self.S0 = self.covid_data.N - self.I0

        # SIRモデル
        self.sir_model_non_normalized = DifferentialEquation(
            func=self.SIR_non_normalized,
            times=self.time_range[1:],
            n_states=2,
            n_theta=2,
            t0=0)

    def run_SIR_model(self, n_samples, n_tune, likelihood, prior):
        # ------------------------- メタデータ --------------------------------#
        now = dt.now()
        timenow = now.strftime("%d-%m-%Y_%H:%M:%S")
        self.filename = 'sir_' + self.covid_data.data_begin.replace('/','-') + '_' + \
            self.covid_data.data_end.replace('/','-') + '_' + timenow
        self.likelihood = likelihood
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.likelihood = likelihood
        self.prior = prior
        # ------------------------ モデルの実行中にメタデータを書き出します -------------------#
        metadata_db_filename = 'metadata_db.db'

        t = time.time()

        with pm.Model() as model4:
            sigma = pm.HalfCauchy('sigma', likelihood['sigma'], shape=1)
            lam = pm.Lognormal('lambda', prior['lam'], prior['lambda_std'])
            mu = pm.Lognormal('mu', prior['mu'], prior['mu_std'])

            res = self.sir_model_non_normalized(y0=[self.S0, self.I0], theta=[lam, mu])

            if(likelihood['distribution'] == 'lognormal'):
                Y = pm.Lognormal('Y', mu=pm.math.log(res.take(0, axis=1)), sigma=sigma, observed=self.covid_data.cases_obs[1:])
            else:
                Y = pm.StudentT( "Y",  nu=likelihood['nu'],       # データの尤度分布
                        mu=res.take(0, axis=1),     # 尤度分布の平均値、これらはSIRの予測となります
                        sigma=sigma,
                        observed=cases_obs[1:]
                        )
            trace = pm.sample(self.n_samples, tune=self.n_tune, target_accept=0.9, cores=2) # コア数指定
            data = az.from_pymc3(trace=trace)

        t1 = time.time() - t
        az.plot_posterior(data, round_to=2, credible_interval=0.95)
        axes = az.plot_trace(trace)
        fig = axes.ravel()[0].figure
        fig.savefig(self.filename)

        self.metadata_db = shelve.open(metadata_db_filename)
        self.metadata_db[self.filename] = {'type': 'sir', 'samples': n_samples,
                                    'tune': n_tune,
                                    'elapsed_time': t1,
                                    'finished': dt.now().strftime("%d-%m-%Y_%H:%M:%S"),
                                    'likelihood': likelihood,
                                    'prior': prior }
        self.metadata_db.close()

# COMMAND ----------

# MAGIC %md
# MAGIC sunodeモジュールを用いるケース
# MAGIC 
# MAGIC **注意** 84行目の`cores=24`の値をお使いのクラスターのコア数に合わせてください

# COMMAND ----------

class SIR_model_sunode():

    def __init__(self, covid_data) :

        # ------------------------- Covid_data オブジェクト -----------------------#
        self.covid_data = covid_data
        # ------------------------- SIRモデルのセットアップ、しかし、実行する際には明示的に呼び出す必要があります ------------------------#
        self.setup_SIR_model()

    def SIR_sunode(self, t, y, p):
        return {
            'S': -p.lam * y.S * y.I,
            'I': p.lam * y.S * y.I - p.mu * y.I,
        }

    def setup_SIR_model(self):
        self.time_range = np.arange(0,len(self.covid_data.cases_obs),1)
        self.I0 = self.covid_data.cases_obs[0]
        self.S0 = self.covid_data.N - self.I0
        self.S_init = self.S0 / self.covid_data.N
        self.I_init = self.I0 / self.covid_data.N
        self.cases_obs_scaled = self.covid_data.cases_obs / self.covid_data.N


    def run_SIR_model(self, n_samples, n_tune, likelihood, prior):
        # ------------------------- メタデータ --------------------------------#
        now = dt.now()
        timenow = now.strftime("%d-%m-%Y_%H:%M:%S")
        self.filename = 'sir_' + self.covid_data.data_begin.replace('/','-') + '_' + \
            self.covid_data.data_end.replace('/','-') + '_' + timenow
        self.likelihood = likelihood
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.likelihood = likelihood
        self.prior = prior
        # ------------------------ モデルの実行中にメタデータを書き出します -------------------#
        metadata_db_filename = 'metadata_db.db'

        t = time.time()

        with pm.Model() as model4:
            sigma = pm.HalfCauchy('sigma', self.likelihood['sigma'], shape=1)
            lam_mu = np.log(self.prior['lam']) + self.prior['lambda_std']**2
            mu_mu = np.log(self.prior['mu']) + self.prior['mu_std']**2
            lam = pm.Lognormal('lambda', lam_mu , self.prior['lambda_std']) # 1.5, 1.5
            mu = pm.Lognormal('mu', mu_mu, self.prior['mu_std'])           # 1.5, 1.5

            res, _, problem, solver, _, _ = sunode.wrappers.as_theano.solve_ivp(
            y0={
    	    # ODEの初期条件、それぞれの変数はtheanoあるいはnumpy変数とshapeを指定する必要があります。
    	    # このdictはネストできます。
                'S': (self.S_init, ()),
                'I': (self.I_init, ()),},
            params={
    	    # ODEのパラメーターです。sunodeはtheano変数の微分のみを計算します。ここでもshapeを指定する必要があります。
            # numpy変数は自動的に推定されます。
    	    # このdictはネストできます。
                'lam': (lam, ()),
                'mu': (mu, ()),
                '_dummy': (np.array(1.), ())},
            # sympy変数を用いてODEの右辺を計算する関数です。
            rhs=self.SIR_sunode,
            # ソリューションにアクセスしたいタイムポイントです。
            tvals=self.time_range,
            t0=self.time_range[0]
            )
            if(likelihood['distribution'] == 'lognormal'):
                I = pm.Lognormal('I', mu=res['I'], sigma=sigma, observed=self.cases_obs_scaled)
            elif(likelihood['distribution'] == 'normal'):
                I = pm.Normal('I', mu=res['I'], sigma=sigma, observed=self.cases_obs_scaled)
            elif(likelihood['distribution'] == 'students-t'):
                I = pm.StudentT( "I",  nu=likelihood['nu'],       # データの尤度分布
                        mu=res['I'],     # 尤度分布の平均値、これらはSIRの予測となります
                        sigma=sigma,
                        observed=self.cases_obs_scaled
                        )

            theano.printing.Print('S')(res['S'])
            print('Problem',problem)
            print('Solver',solver)

            R = 1 - (res['I'] + res['S'])
            #S = 1 - (res['I'][1:])
            #theano.printing.Print('R')(R)
            R0 = pm.Deterministic('R0',lam/mu)

            step = pm.Metropolis()
            trace = pm.sample(self.n_samples, tune=self.n_tune, chains=4, cores=24) # コア数指定
            data = az.from_pymc3(trace=trace)

        t1 = time.time() - t
        az.plot_posterior(data, round_to=2, point_estimate='mode', credible_interval=0.95)
        axes = az.plot_trace(trace)
        fig = axes.ravel()[0].figure
        fig.savefig(self.filename)
        
        fig = ff.create_distplot([trace['R0']], bin_size=0.5, group_labels=['x'])

        # タイトルの追加
        fig.update_layout(title_text='R0(基本再生産数)の曲線、ラグプロット')
        fig.update_xaxes(range=[0,7])
              

        self.metadata_db = shelve.open(metadata_db_filename)
        self.metadata_db[self.filename] = {'type': 'sir', 'samples': n_samples,
                                    'tune': n_tune,
                                    'elapsed_time': t1,
                                    'finished': dt.now().strftime("%d-%m-%Y_%H:%M:%S"),
                                    'likelihood': likelihood,
                                    'prior': prior }
        self.metadata_db.close()
        return(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## SIRモデルによるパラメーター推定、R0の予測

# COMMAND ----------

# MAGIC %md
# MAGIC ### 日本
# MAGIC 
# MAGIC 以下の例では2021/1/1から5/31の期間のデータで推定、予測を行っています。

# COMMAND ----------

covid_obj = COVID_data('Japan', Population=126.3e6)
covid_obj.get_dates(data_begin='1/1/21', data_end='5/31/21')

# sunodeを使用
sir_model = SIR_model_sunode(covid_obj)
# 尤度分布
likelihood = {'distribution': 'lognormal', 
              'sigma': 2}
# 事前確率
prior = {'lam': 1.0, 
         'mu': 0.5, 
         'lambda_std': 1.0,
         'mu_std': 0.2 }
fig1 = sir_model.run_SIR_model(n_samples=2000, n_tune=1000, likelihood=likelihood, prior=prior)

# COMMAND ----------

# MAGIC %md
# MAGIC R0(基本再生産数)の分布をヒストグラム、曲線、ラグプロット形式で表示

# COMMAND ----------

fig1.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### アメリカ

# COMMAND ----------

covid_obj = COVID_data('US', Population=328.2e6)
covid_obj.get_dates(data_begin='1/1/21', data_end='5/31/21')

# sunodeを使用
sir_model = SIR_model_sunode(covid_obj)
# 尤度分布
likelihood = {'distribution': 'lognormal', 
              'sigma': 2}
# 事前確率
prior = {'lam': 1.0, 
         'mu': 0.5, 
         'lambda_std': 1.0,
         'mu_std': 0.2 }

# R0(基本再生産数)の分布をヒストグラム、曲線、ラグプロット形式で取得
fig1 = sir_model.run_SIR_model(n_samples=2000, n_tune=1000, likelihood=likelihood, prior=prior)

# COMMAND ----------

# MAGIC %md
# MAGIC R0(基本再生産数)の分布をヒストグラム、曲線、ラグプロット形式で表示

# COMMAND ----------

fig1.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # END
