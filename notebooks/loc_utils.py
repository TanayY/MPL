import pandas as pd


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
