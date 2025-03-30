import luigi
from typing_extensions import Annotated
import typer
from erisk.etl.embedding.preprocess_utils import PreprocessSelfReferentialPosts
from ml import WrappedSentenceTransformer


class PreprocessAndEmbedWorkflow(luigi.WrapperTask):
    raw_path = luigi.Parameter()
    preprocessed_path = luigi.Parameter()
    embeddings_path = luigi.Parameter()
    model_name = luigi.Parameter(default="all-MiniLM-L6-v2")
    batch_size = luigi.IntParameter(default=16)

    def requires(self):
        yield PreprocessSelfReferentialPosts(
            input_path=self.raw_path,
            output_path=self.preprocessed_path,
        )
        yield WrappedSentenceTransformer(
            input_path=self.preprocessed_path,
            output_path=self.embeddings_path,
            model_name=self.model_name,
            batch_size=self.batch_size,
        )


def main(
    raw_path: Annotated[str, typer.Argument(help="Path to raw Reddit data")],
    preprocessed_path: Annotated[
        str, typer.Argument(help="Path to save preprocessed output")
    ],
    embeddings_path: Annotated[
        str, typer.Argument(help="Path to save sentence embeddings")
    ],
    model_name: Annotated[str, typer.Option()] = "all-MiniLM-L6-v2",
    batch_size: Annotated[int, typer.Option()] = 16,
    scheduler_host: Annotated[str, typer.Option()] = None,
):
    kwargs = {}
    if scheduler_host:
        kwargs["scheduler_host"] = scheduler_host
    else:
        kwargs["local_scheduler"] = True

    luigi.build(
        [
            PreprocessAndEmbedWorkflow(
                raw_path=raw_path,
                preprocessed_path=preprocessed_path,
                embeddings_path=embeddings_path,
                model_name=model_name,
                batch_size=batch_size,
            )
        ],
        **kwargs,
    )


if __name__ == "__main__":
    typer.run(main)
