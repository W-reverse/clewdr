use bytes::Bytes;
use eventsource_stream::Eventsource;
use futures::Stream;
use std::pin::Pin;

pub type RawEventStream = eventsource_stream::EventStream<Pin<Box<dyn Stream<Item = Result<Bytes, wreq::Error>> + Send + 'static>>>;

pub fn wrap_response_stream(resp: wreq::Response) -> RawEventStream {
    let bytes_stream = resp.bytes_stream();
    let boxed: Pin<Box<dyn Stream<Item = Result<Bytes, wreq::Error>> + Send>> = Box::pin(bytes_stream);
    boxed.eventsource()
}
