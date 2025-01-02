import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# model evaluation
from sklearn.metrics import mean_squared_error


class utilities:
    import pandas as pd

    def aggregate_by_state(
        statename, min_year, max_year, agg_enquiry, agg_booking, agg_invoice
    ):

        start_day = pd.Timestamp(f"{min_year}-01-01")
        end_day = pd.Timestamp(f"{max_year}-12-01")
        date_range = pd.date_range(start_day, end_day, freq="MS")  # MS = month start

        # For Enquiry
        agg_enquiry_by_state = agg_enquiry[agg_enquiry["StateName"] == statename].copy()

        # Check if EnquiryMonth is Period type, and convert if necessary
        if agg_enquiry_by_state["EnquiryMonth"].dtype.name == "period[M]":
            agg_enquiry_by_state["EnquiryMonth"] = agg_enquiry_by_state[
                "EnquiryMonth"
            ].dt.to_timestamp()

        # Set EnquiryMonth as index and reindex with the full date range
        agg_enquiry_by_state = agg_enquiry_by_state.set_index("EnquiryMonth").reindex(
            date_range, fill_value=0
        )

        # For Booking
        agg_booking_by_state = agg_booking[agg_booking["StateName"] == statename].copy()

        # Check if BookingMonth is Period type, and convert if necessary
        if agg_booking_by_state["BookingMonth"].dtype.name == "period[M]":
            agg_booking_by_state["BookingMonth"] = agg_booking_by_state[
                "BookingMonth"
            ].dt.to_timestamp()

        # Set BookingMonth as index and reindex with the full date range
        agg_booking_by_state = agg_booking_by_state.set_index("BookingMonth").reindex(
            date_range, fill_value=0
        )

        # For Invoice
        agg_invoice_by_state = agg_invoice[agg_invoice["StateName"] == statename].copy()

        # Check if InvoiceMonth is Period type, and convert if necessary
        if agg_invoice_by_state["InvoiceMonth"].dtype.name == "period[M]":
            agg_invoice_by_state["InvoiceMonth"] = agg_invoice_by_state[
                "InvoiceMonth"
            ].dt.to_timestamp()

        # Set InvoiceMonth as index and reindex with the full date range
        agg_invoice_by_state = agg_invoice_by_state.set_index("InvoiceMonth").reindex(
            date_range, fill_value=0
        )

        return agg_enquiry_by_state, agg_booking_by_state, agg_invoice_by_state

    def model_evaluation(model, Xtrain, ytrain, Xtest, ytest):
        def fit_scatter_plot(X, y, set_name):
            y_fitted_values = model.predict(X)
            xmin = y.min()
            xmax = y.max()
            plt.scatter(x=y_fitted_values, y=y, alpha=0.25)
            x_line = np.linspace(xmin, xmax, 10)
            y_line = x_line
            plt.plot(x_line, y_line, "r--")
            plt.axhline(0, color="black", linestyle="--")
            plt.xlabel("Prediction")
            plt.ylabel("True Value")
            plt.title(f"Plot of predicted values versus true values - {set_name} set")

        def plot_of_residuals(X, y, set_name):
            errors = model.predict(X) - np.reshape(np.array(y), (-1))
            plt.scatter(x=y, y=errors, alpha=0.25)
            plt.axhline(0, color="r", linestyle="--")
            plt.xlabel("True Value")
            plt.ylabel("Residual")
            plt.title(f"Plot of residuals - {set_name} set")

        def hist_of_residuals(X, y, set_name):
            errors = model.predict(X) - np.reshape(np.array(y), (-1))
            plt.hist(errors, bins=100)
            plt.axvline(errors.mean(), color="k", linestyle="dashed", linewidth=1)
            plt.title(f"Histogram of residuals - {set_name} set")

        def DPA(y_true, y_pred):
            dpa = 100 - (((np.sum(np.abs(y_pred - y_true))) / (np.sum(y_true))) * 100)
            return dpa

        def BIAS(y_true, y_pred):
            bias = ((np.sum(y_pred - y_true)) / (np.sum(y_true))) * 100
            return bias

        fig = plt.figure(figsize=(16, 10))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        ax = fig.add_subplot(3, 2, 1)
        fit_scatter_plot(X=Xtrain, y=ytrain, set_name="train")

        ax = fig.add_subplot(3, 2, 2)
        fit_scatter_plot(X=Xtest, y=ytest, set_name="test")

        ax = fig.add_subplot(3, 2, 3)
        plot_of_residuals(X=Xtrain, y=ytrain, set_name="train")

        ax = fig.add_subplot(3, 2, 4)
        plot_of_residuals(X=Xtest, y=ytest, set_name="test")

        ax = fig.add_subplot(3, 2, 5)
        hist_of_residuals(X=Xtrain, y=ytrain, set_name="train")

        ax = fig.add_subplot(3, 2, 6)
        hist_of_residuals(X=Xtest, y=ytest, set_name="test")

        plt.show()

        y_pred_train = model.predict(Xtrain)
        print(f"RMSE train: {sqrt(mean_squared_error(ytrain, y_pred_train))}")
        print(f"DPA  train: {DPA(ytrain, y_pred_train)}")
        print(f"BIAS train: {BIAS(ytrain, y_pred_train)}")
        print()
        y_pred_test = model.predict(Xtest)
        print(f"RMSE test:  {sqrt(mean_squared_error(ytest, y_pred_test))}")
        print(f"DPA  test: {DPA(ytest, y_pred_test)}")
        print(f"BIAS test: {BIAS(ytest, y_pred_test)}")

        # Feature Importance
        importance = model.feature_importances_
        df_feature_importance = importance.argsort()
        df_feature_importance = pd.DataFrame(
            {
                "column": Xtrain.columns[df_feature_importance],
                "importance": importance[df_feature_importance],
            }
        )
        df_feature_importance = (
            df_feature_importance[df_feature_importance["importance"] >= 0.01]
            .copy()
            .reset_index(drop=True)
        )
        plt.figure(figsize=(16, 4))
        plt.barh(
            df_feature_importance["column"][-10:],
            df_feature_importance["importance"][-10:],
        )
        plt.tick_params(axis="both", labelsize=10)
        plt.title("Model Feature Importance", size=20)
        plt.xlabel(" ", size=15)
        plt.tight_layout()
        plt.show()

    def plot_true_pred(df, big_prod, number, target, type_pred, horizon, scenario):
        # df.index = pd.to_datetime(df.index, errors="coerce")
        # start_date = pd.to_datetime(df.index[0], format="%Y-%m-%d")
        # df.index = pd.period_range(start=start_date, periods=len(df), freq="M")
        # df = df.index.to_timestamp()
        # print("Index type is: ", type(df.index))
        for idx in range(big_prod.shape[0])[:number]:
            prod = (df[["StateName", "KVA_Id"]] == big_prod.iloc[idx, :2]).all(axis=1)
            fig = plt.figure(figsize=(16, 2))

            df_for_plotting = df.copy()
            df_for_plotting.index = df.index.to_timestamp()

            ax1 = fig.add_subplot(111)
            ax1.plot(
                df_for_plotting.index[prod],
                df.loc[prod, target],
                color="green",
                label=type_pred,
                linewidth=3,
            )
            ax1.plot(
                df_for_plotting.index[prod],  # .shift(-horizon),
                df.loc[prod, f"Prediction_{type_pred}_{horizon}month"],
                color="black",
                label=f"{type_pred} -{horizon}month",
                linewidth=3,
            )
            ### Add promo if Scenario 1 or 3 ##
            # if (scenario==1)|(scenario==2):
            #     ax1.scatter(df.loc[(prod)&(df['type_promo_1']==1), 'PERIOD_TAG'],
            #                 df.loc[(prod)&(df['type_promo_1']==1), target],
            #                 label='type_promo_1', alpha=0.5, s=100)
            #     ax1.scatter(df.loc[(prod)&(df['type_promo_2']==1), 'PERIOD_TAG'],
            #                 df.loc[(prod)&(df['type_promo_2']==1), target],
            #                 label='type_promo_2', alpha=0.5, s=100)
            #     ax2 = ax1.twinx()
            #     ax2.plot(df.loc[prod, 'PERIOD_TAG'], df.loc[prod, 'numeric_distribution_selling_promotion'],
            #             color='blue', label='Promo Distribution')
            #     ax2.set_ylim([0, 100])
            #     ax2.legend(loc='upper right')

            # ax1.set_xlim([df.loc[((prod)&(df['PERIOD_TAG']>='2019-01-01')), 'PERIOD_TAG'].min(),
            #             df.loc[((prod)&(df['PERIOD_TAG']<'2020-01-01')), 'PERIOD_TAG'].max()])
            # ax1.set_ylim(ymin=0)
            ax1.legend(loc="upper left")
            plt.title(
                f"State: {big_prod.iloc[idx, 0]}, Product: {big_prod.iloc[idx, 1]}"
            )
            plt.show()
