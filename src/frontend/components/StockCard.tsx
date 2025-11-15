import { Card } from "@/frontend/components/ui/card";
import { TrendingUp, TrendingDown, Minus, Clock } from "lucide-react";
import { Badge } from "@/frontend/components/ui/badge";

interface StockCardProps {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  prediction?: {
    trend: "up" | "down" | "neutral";
    confidence: number;
    predictedChange: number;
  };
  volume: number;
  high: number;
  low: number;
  open: number;
  isMarketOpen: boolean;
}

export const StockCard = ({
  symbol,
  name,
  price,
  change,
  changePercent,
  prediction,
  volume,
  high,
  low,
  open,
  isMarketOpen,
}: StockCardProps) => {
  const isPositive = change >= 0;
  const TrendIcon = isPositive ? TrendingUp : TrendingDown;

  const getPredictionIcon = () => {
    if (!prediction) return Minus;
    switch (prediction.trend) {
      case "up":
        return TrendingUp;
      case "down":
        return TrendingDown;
      default:
        return Minus;
    }
  };

  const PredictionIcon = getPredictionIcon();

  return (
    <Card className="p-6 bg-card border-border hover:shadow-lg transition-all duration-300">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-2xl font-bold text-foreground">{symbol}</h3>
          <p className="text-sm text-muted-foreground">{name}</p>
        </div>
        <Badge variant={isMarketOpen ? "default" : "secondary"} className="flex items-center gap-1">
          <Clock className="h-3 w-3" />
          {isMarketOpen ? "Live" : "Closed"}
        </Badge>
      </div>

      <div className="space-y-4">
        <div>
          <div className="text-3xl font-bold text-foreground mb-1">${price.toFixed(2)}</div>
          <div
            className={`flex items-center gap-2 ${
              isPositive ? "text-success" : "text-destructive"
            }`}
          >
            <TrendIcon className="h-4 w-4" />
            <span className="font-semibold">
              {isPositive ? "+" : ""}
              {change.toFixed(2)} ({isPositive ? "+" : ""}
              {changePercent.toFixed(2)}%)
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <span className="text-muted-foreground">Open:</span>
            <span className="ml-2 text-foreground font-medium">${open.toFixed(2)}</span>
          </div>
          <div>
            <span className="text-muted-foreground">High:</span>
            <span className="ml-2 text-foreground font-medium">${high.toFixed(2)}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Low:</span>
            <span className="ml-2 text-foreground font-medium">${low.toFixed(2)}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Volume:</span>
            <span className="ml-2 text-foreground font-medium">
              {(volume / 1000000).toFixed(2)}M
            </span>
          </div>
        </div>

        {prediction && (
          <div
            className={`p-4 rounded-lg border ${
              prediction.trend === "up"
                ? "bg-success/10 border-success/30"
                : prediction.trend === "down"
                ? "bg-destructive/10 border-destructive/30"
                : "bg-muted border-border"
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-foreground">Trained Model Prediction</span>
              <div
                className={`flex items-center gap-1 ${
                  prediction.trend === "up"
                    ? "text-success"
                    : prediction.trend === "down"
                    ? "text-destructive"
                    : "text-muted-foreground"
                }`}
              >
                <PredictionIcon className="h-4 w-4" />
                <span className="font-semibold">
                  {prediction.predictedChange > 0 ? "+" : ""}
                  {prediction.predictedChange.toFixed(2)}%
                </span>
              </div>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">Next 5-15 min</span>
              <span className="text-muted-foreground">
                Confidence: {(prediction.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <div className="mt-2 h-1.5 bg-secondary rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${
                  prediction.trend === "up" ? "bg-success" : "bg-destructive"
                }`}
                style={{ width: `${prediction.confidence * 100}%` }}
              />
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};
