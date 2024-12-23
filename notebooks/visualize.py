import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loc_utils import utilities

# import os


class visualize:
    @staticmethod
    def countplot_kva(df):
        plt.figure(figsize=(10, 6))
        sns.countplot(x="KVA_Id", data=df, order=df["KVA_Id"].value_counts().index)
        plt.xticks(rotation=90)
        plt.title("Count of Each KVA ID")
        plt.xlabel("KVA ID")
        plt.ylabel("Count")
        plt.show()

    @staticmethod
    def plot_by_state(statename, agg_enquiry, agg_booking, agg_invoice):
        min_year = min(
            agg_enquiry[agg_enquiry["StateName"] == statename]["EnquiryMonth"]
        ).year
        max_year = max(
            agg_enquiry[agg_enquiry["StateName"] == statename]["EnquiryMonth"]
        ).year
        agg_enquiry_by_state, agg_booking_by_state, agg_invoice_by_state = (
            utilities.aggregate_by_state(
                statename, min_year, max_year, agg_enquiry, agg_booking, agg_invoice
            )
        )

        for j in range(min_year, max_year + 1):

            plot_for_enquiry = agg_enquiry_by_state[
                agg_enquiry_by_state.index.year == j
            ].copy()
            plot_for_booking = agg_booking_by_state[
                agg_booking_by_state.index.year == j
            ].copy()
            plot_for_invoice = agg_invoice_by_state[
                agg_invoice_by_state.index.year == j
            ].copy()
            # plot_end = agg_enquiry_by_state[agg_enquiry_by_state.index.year == j].idxmax()

            # print("Plotting range: ", plot_start, ":", plot_end)

            plt.figure(figsize=(12, 6), dpi=300)

            # Plot enquiry time series
            plt.plot(
                plot_for_enquiry[["Enquiry_Quantity"]].index.astype(str),
                plot_for_enquiry[["Enquiry_Quantity"]],
                label="Enquiries",
                color="blue",
                linewidth=3,
                alpha=0.7,
            )

            # Plot booking time series
            plt.plot(
                plot_for_booking[["Booking_Quantity"]].index.astype(str),
                plot_for_booking[["Booking_Quantity"]],
                label="Bookings",
                color="green",
                linewidth=2,
                alpha=0.7,
            )

            # Plot invoice time series
            plt.plot(
                plot_for_invoice[["Invoice_Quantity"]].index.astype(str),
                plot_for_invoice[["Invoice_Quantity"]],
                label="Invoices",
                color="red",
                linewidth=1,
                alpha=0.7,
            )

            # Adding labels and title
            plt.xlabel("Year-Month")
            plt.ylabel("Quantity")
            plt.title(
                f"Time Series of Enquiries, Bookings, and Invoices (Before Forecasting) - {statename} ({j})"
            )
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            save_path = rf"C:\Users\Tanay Yeole\Documents\My_Projects\MahindraPowerol\reports\figures\{statename}\P5_{j}.png"
            directory = os.path.dirname(save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(save_path, dpi=300)
            plt.close()

        # --------------------------------------------------------------------------------------------------------------------
        # Overall plot for the state
        plt.figure(figsize=(12, 6), dpi=300)

        # Plot enquiry time series
        plt.plot(
            agg_enquiry_by_state[["Enquiry_Quantity"]].index.astype(str),
            agg_enquiry_by_state[["Enquiry_Quantity"]],
            label="Enquiries",
            color="blue",
            linewidth=3,
            alpha=0.7,
        )

        # Plot booking time series
        plt.plot(
            agg_booking_by_state[["Booking_Quantity"]].index.astype(str),
            agg_booking_by_state[["Booking_Quantity"]],
            label="Bookings",
            color="green",
            linewidth=2,
            alpha=0.7,
        )

        # Plot invoice time series
        plt.plot(
            agg_invoice_by_state[["Invoice_Quantity"]].index.astype(str),
            agg_invoice_by_state[["Invoice_Quantity"]],
            label="Invoices",
            color="red",
            linewidth=1,
            alpha=0.7,
        )

        # Adding labels and title
        plt.xlabel("Year-Month")
        plt.ylabel("Quantity")
        plt.title(
            f"Overall Time Series of Enquiries, Bookings, and Invoices (Before Forecasting) - {statename}"
        )
        plt.legend()
        plt.xticks(rotation=90)
        plt.tight_layout()
        save_path = rf"C:\Users\Tanay Yeole\Documents\My_Projects\MahindraPowerol\reports\figures\{statename}\overall.png"
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, dpi=300)
        plt.close()
